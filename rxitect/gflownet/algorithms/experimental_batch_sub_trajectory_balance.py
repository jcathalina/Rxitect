from typing import Any, Dict

import numpy as np
import torch
import torch_geometric.data as gd
import math
from torch import nn
from torch_scatter import scatter

from rxitect.gflownet.algorithms.trajectory_balance import TrajectoryBalance
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.interfaces.graph_context import IGraphContext


class SubTrajectoryBalance(TrajectoryBalance):
    """
    Class that inherits from TrajectoryBalance and implements the sub-trajectory balance (lambda) algorithm
    """
    def __init__(self, env: GraphBuildingEnv, ctx: IGraphContext, rng: np.random.RandomState,
                 hps: Dict[str, Any], max_len=None, max_nodes=None, lambd: float = 1.0):
        super().__init__(env, ctx, rng, hps, max_len, max_nodes)
        self.lambd = lambd

    def create_training_data_from_own_samples(self, model: nn.Module, n: int, cond_info: torch.Tensor):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model:
           The model being sampled
        n:
            Number of samples to create
        cond_info: Tensor
            Conditional information, shape (N, n_info)

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, dev)
        return data

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute sub-trajectory (lambda) loss over trajectories contained in the batch"""
        dev = batch.x.device
        # A single trajectory comprises multiple graphs
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = batch.rewards
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)

        # Forward pass of the model, returns a GraphActionCategorical and the optional bootstrap predictions
        fwd_cat, logF = model(batch, cond_info[batch_idx])

        # This is the log prob of each action in the trajectory
        logP_F = fwd_cat.log_prob(batch.actions)
        # The log prob of each backward action
        logP_B = (1 / batch.num_backward).log()
        # Take log rewards, and clip
        Rp = torch.maximum(rewards.log(), torch.tensor(-100.0, device=dev))
        # This is the log probability of each trajectory

        max_num_sub_trajectories = math.comb(max(batch.traj_lens) + 1, 2)
        lst_lst = _generate_sub_traj_batch_index_ranges(batch.traj_lens)
        final_stuff = _bundle_sub_traj_batch_index_ranges(batch.traj_lens, lst_lst)
        Q = _wow(batch_idx, num_trajs, final_stuff)
        total_traj_prob = 0
        total_traj_prob_backwards = 0
        for q in range(max_num_sub_trajectories):
            # take num trajs + 1 to account for dummy dimension, and then leave sum of dummy dimension logp out
            total_traj_prob += scatter(logP_F, torch.from_numpy(Q[q]), dim=0, dim_size=num_trajs + 1)[:-1]
            total_traj_prob_backwards += scatter(logP_B, torch.from_numpy(Q[q]), dim=0, dim_size=num_trajs + 1)[:-1]
            # [A], [A, B], [A, B, C]....[B, C]
            # logF[A] + scatter([A])
            # logF[A] + scatter([A, B])
            # logF[A] + scatter([A, B, C])
            # ...
            # logF[B] + scatter([B, C])
            # logF[what_i_want_1] + logF[what_i_want_2] # TODO: Figure out how we can get the first index for each subtrajectory in the batch of subtraj indices Q[q] :)

        # numerator = torch.sum(logF[]) + total_traj_prob
        # denominator = torch.sum(logF[]) + total_traj_prob_backwards
        traj_losses = (numerator - denominator).pow(2)
        loss = traj_losses.mean()
        info = {
            'loss': loss.item(),
        }
        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return loss, info


def _generate_sub_traj_batch_index_ranges(traj_lens):
    lst_lst = []
    cumlens = np.hstack([np.array([0]), np.cumsum(traj_lens)])
    nT = len(traj_lens)
    for n in range(nT):
        offset = cumlens[n]
        l = traj_lens[n]
        lst = []
        for i in range(l):
            for j in range(i, l):
                lst.append([offset+i, offset+j])
        lst_lst.append(lst)
    return lst_lst


def _bundle_sub_traj_batch_index_ranges(traj_lens, lst_lst):
    max_traj_possible = math.comb(max(traj_lens) + 1, 2)  # n+1 choose 2 possible sub-trajectories.
    final_stuff = []
    nT = len(traj_lens)
    for i in range(max_traj_possible):
        stuff = []
        for j in range(nT):
            try:
                x = lst_lst[j][i]
                stuff.append(x)
            except IndexError:
                continue
        final_stuff.append(stuff)
    return final_stuff


def _wow(T, nT, final_stuff):
    aaaa = []
    for fstuff in final_stuff:
        test = np.full_like(T, fill_value=-1)
        for fs in fstuff:
            left = fs[0]
            right = fs[1] + 1
            test[left:right] = 1
        aaaa.append(test)
    Q = (aaaa + 1 * T) - 1
    Q[Q < 0] = nT
    return Q
