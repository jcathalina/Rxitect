from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter

from rxitect.gflownet.algorithms.interfaces.graph_algorithm import IGraphAlgorithm
from rxitect.gflownet.algorithms.trajectory_balance import TrajectoryBalance
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.interfaces.graph_context import IGraphContext
from rxitect.gflownet.utils.graph import count_backward_transitions
from rxitect.gflownet.utils.graph_sampler import GraphSampler


@torch.jit.script
def sub_tb_lambda_loss(traj_lens, log_prob, log_p_B, log_reward_preds, Rp, invalid_mask, LAMBDA: float = 1.0, dev: str = "cpu"
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT implementation of the Sub-trajectory Balance (Lambda) loss function, as implemented in
    https://github.com/GFNOrg/gflownet/blob/subtb/mols/gflownet.py,
    by Malkin et al., 2022. @ https://arxiv.org/pdf/2209.12782.pdf

    Parameters
    ----------
    traj_lens:
        The lengths of the sampled trajectories
    log_prob:
        The forward policy ($P_F$), containing log probs of all forward actions in the trajectories
    log_p_B:
        The backward policy ($P_B$), containing log probs of all backward transitions in the trajectories
    log_reward_preds:
        The flow predictions ($F$) for each (partial) trajectory
        (i.e., the sum of flows into the final node of that particular trajectory, with the terminal node's
        flow being equal to the reward assigned to the molecule)
    Rp:
        The rewards associated with each completed trajectory
    invalid_mask:
        Tensor containing masks where the trajectory is illegal
    LAMBDA:
        Hyperparam controlling weights assigned to subtrajectories of different lengths.
        Setting this to 1.0 (default) results in a uniform weighting scheme for all subjtractory lengths.
        As this approaches infinity, the weight is almost only purely assigned to the completed trajectory,
        resulting in the original Trajectory Balance objective.
    dev:
        The device to process the given data on ('cpu' or 'cuda' for GPU)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the total trajectory losses and the total accumulated lambda, respectively.
    """
    num_trajs = int(traj_lens.shape[0])
    traj_losses = torch.zeros(1, device=dev)
    total_LAMBDA = torch.zeros(1, device=dev)

    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=dev), traj_lens]), 0).long()
    for ep in range(num_trajs):
        offset = cumul_lens[ep]
        T = int(traj_lens[ep])
        for i in range(T):
            for j in range(i, T):
                flag = float(j + 1 < T)
                acc = log_reward_preds[offset + i] - log_reward_preds[offset + min(j + 1, T - 1)] * flag - Rp[ep] * (1 - flag)
                for k in range(i, j + 1):
                    numerator = log_prob[offset + k]
                    denominator = log_p_B[offset + k]
                    denominator = denominator * (1 - invalid_mask[ep]) + invalid_mask[ep] * (numerator.detach() - 1)
                    # acc += log_prob[offset + k] - log_p_B[offset + k]  # This is the SubTB loss, before squaring
                    acc += numerator - denominator
                traj_losses += acc.pow(2) * LAMBDA ** (j - 1 + 1)  # SubTB loss * Total lambda for sub-traj. i:j
                total_LAMBDA += LAMBDA ** (j - 1 + 1)  # denominator
    return traj_losses, total_LAMBDA


class SubTrajectoryBalance(TrajectoryBalance):
    """
    """

    def __init__(self, env: GraphBuildingEnv, ctx: IGraphContext, rng: np.random.RandomState, hps: Dict[str, Any],
                 max_len: Optional[int] = None, max_nodes: Optional[int] = None) -> None:
        """TB implementation, see
        "Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
        Emmanuel Bengio, Chen Sun, Yoshua Bengio"
        https://arxiv.org/abs/2201.13259

        Hyperparameters used:
        random_action_prob: float, probability of taking a uniform random action when sampling
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions
        bootstrap_own_reward: bool, if True, uses the .reward batch data to predict rewards for sampled data
        tb_epsilon: float, if not None, adds this epsilon in the numerator and denominator of the log-ratio
        reward_loss_multiplier: float, multiplying constant for the bootstrap loss.

        Parameters
        ----------
        env:
            A graph building environment.
        ctx:
            A context used to build graphs with.
        rng:
            random state to sample random actions from
        hps: Dict[str, Any]
            Hyperparameter dictionary, see above for used keys.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        """
        super().__init__(env, ctx, rng, hps, max_len, max_nodes)
        self.ctx = ctx
        self.env = env
        self.rng = rng
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.illegal_action_logreward = hps['illegal_action_logreward']
        self.bootstrap_own_reward = hps['bootstrap_own_reward']
        self.epsilon = hps['tb_epsilon']
        self.reward_loss_multiplier = hps['reward_loss_multiplier']
        # Experimental flags
        self.reward_loss_is_mae = True
        self.tb_loss_is_mae = False
        self.tb_loss_is_huber = False
        self.mask_invalid_rewards = False
        self.length_normalize_losses = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.graph_sampler = GraphSampler(ctx, env, max_len, max_nodes, rng, self.sample_temp)
        self.graph_sampler.random_action_prob = hps['random_action_prob']
        self.LAMBDA = hps.get('lambda', 1.0)

    def create_training_data_from_own_samples(self, model: nn.Module, n: int, cond_info: Tensor):
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

    def compute_batch_losses(self, model: nn.Module, batch: Batch, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model:
            A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
            Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch:
            Batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap:
            the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory comprises many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = batch.rewards
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)  # idcs in original
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and the optional bootstrap predictions
        fwd_cat, log_reward_preds = model(batch, cond_info[batch_idx])

        # Retrieve the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        # log_reward_preds = log_reward_preds[final_graph_idx, 0]

        # This is the log prob of each action in the trajectory
        log_prob = fwd_cat.log_prob(batch.actions)
        # The log prob of each backward action
        log_p_B = (1 / batch.num_backward).log()
        # Take log rewards, and clip
        assert rewards.ndim == 1
        Rp = torch.maximum(rewards.log(), torch.tensor(-100.0, device=dev))
        invalid_mask = 1 - batch.is_valid
        # if self.mask_invalid_rewards:
        #     # Instead of being rude to the model and giving a
        #     # logreward of -100 what if we say, whatever you think the
        #     # logprobablity of this trajetcory is it should be smaller
        #     # (thus the `numerator - 1`). Why 1? Intuition?
        #     denominator = traj_losses * (1 - invalid_mask) + invalid_mask * (total_LAMBDA.detach() - 1)
        traj_losses, total_LAMBDA = sub_tb_lambda_loss(batch.traj_lens, log_prob, log_p_B, log_reward_preds,
                                                       Rp, invalid_mask, self.LAMBDA, str(dev))

        # This is the log probability of each trajectory
        traj_log_prob = scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')  # traj_logits in original
        # # Compute log numerator and denominator of the TB objective
        # numerator = Z + traj_log_prob
        # denominator = Rp + scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')

        # if self.epsilon is not None:
        #     # Numerical stability epsilon
        #     epsilon = torch.tensor([self.epsilon], device=dev).float()
        #     numerator = torch.logaddexp(numerator, epsilon)
        #     denominator = torch.logaddexp(denominator, epsilon)
        #


        # if self.tb_loss_is_mae:
        #     traj_losses = abs(numerator - denominator)
        # else:
        #     traj_losses = (numerator - denominator).pow(2)
        #
        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        # if self.reward_normalize_losses:
        #     # multiply each loss by how important it is, using R as the importance factor
        #     # factor = Rp.exp() / Rp.exp().sum()
        #     factor = -Rp.min() + Rp + 1
        #     factor = factor / factor.sum()
        #     assert factor.shape == traj_losses.shape
        #     # * num_trajs because we're doing a convex combination, and a .mean() later, which would
        #     # undercount (by 2N) the contribution of each loss
        #     traj_losses = factor * traj_losses * num_trajs
        #
        # if self.bootstrap_own_reward:
        #     num_bootstrap = num_bootstrap or len(rewards)
        #     if self.reward_loss_is_mae:
        #         reward_losses = abs(rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp())
        #     else:
        #         reward_losses = (rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp()).pow(2)
        #     reward_loss = reward_losses.mean()
        # else:
        #     reward_loss = 0
        reward_loss = 0

        # loss = traj_losses.mean() + reward_loss * self.reward_loss_multiplier
        loss = traj_losses / total_LAMBDA
        info = {
            'offline_loss': traj_losses[:batch.num_offline] / total_LAMBDA if batch.num_offline > 0 else 0,
            'online_loss': traj_losses[batch.num_offline:] / total_LAMBDA if batch.num_online > 0 else 0,
            'reward_loss': reward_loss,
            'invalid_trajectories': invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            'invalid_logprob': (invalid_mask * traj_log_prob).sum() / (invalid_mask.sum() + 1e-4),
            'invalid_losses': (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            'loss': loss.item(),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')

        return loss, info
