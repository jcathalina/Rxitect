from typing import Tuple

import numpy as np
from numpy.random import RandomState
from torch import nn
import torch

from composable_mol import ComposableMolecule
from fragment_env import FragmentEnv
from gflownet import FragmentGFlowNetOutput
from dataclasses import dataclass


@dataclass(slots=True)
class Trajectory:
    parent_action_pairs: Tuple[ComposableMolecule, Tuple[int, int]]
    reward: float
    curr_mol: ComposableMolecule
    done: bool = False


class Sampler:
    def __init__(self,
                 train_rng: RandomState,
                 sampling_model: nn.Module,
                 env: FragmentEnv,
                 random_action_prob: float = 1e-3,
                 replay_mode: str = "online") -> None:
        self.train_rng = train_rng
        self.sampled_mols = []
        self.sampling_model = sampling_model
        self.env = env
        self.random_action_prob = random_action_prob
        self.early_stop_reg = 0
        self._get_reward = lambda x: 10  # TODO: DUMMY
        self.replay_mode = replay_mode

    def sample_from_model(self, min_fragments: int = 2, max_fragments: int = 8):
        m = ComposableMolecule()
        r = 1e-2  # min reward
        samples = []
        trajectory_stats = []

        if self.early_stop_reg > 0 and np.random.uniform() < self.early_stop_reg:
            early_stop_at = np.random.randint(min_fragments, max_fragments + 1)
        else:
            early_stop_at = max_fragments + 1
        for t in range(max_fragments):
            s = self.env.cmols_to_batch([m])
            output: FragmentGFlowNetOutput = self.sampling_model(s)

            m_o, s_o = output.mol_preds, output.stem_preds
            ## fix from run 330 onwards
            if t < min_fragments:
                m_o = m_o * 0 - 1000  # prevent assigning prob to stop
                # when we can't stop
            ##
            logits = torch.cat([m_o[:, 0].reshape(-1), s_o.reshape(-1)])
            # print(m_o.shape, s_o.shape, logits.shape)
            # print(m.blockidxs, m.jbonds, m.stems)
            cat = torch.distributions.Categorical(
                logits=logits)
            action = cat.sample().item()
            # print(action)
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < min_fragments), logits.shape[0])
            if t == early_stop_at:
                action = 0

            q = torch.cat([m_o[:, 0].reshape(-1), s_o.reshape(-1)])
            trajectory_stats.append((q[action].item(), action, torch.logsumexp(q, 0).item()))
            if t >= min_fragments and action == 0:
                r = self._get_reward(m)
                samples.append(((m,), ((-1, 0),), r, None, 1))
                break
            else:
                action = max(0, action - 1)
                action = (action % self.env.num_frags, action // self.env.num_frags)

                m = self.env.add_fragment(m, *action)
                if len(m.fragments) and not len(m.stems) or t == max_fragments - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r = self._get_reward(m)
                    samples.append((*zip(*self.env.parents(m)), r, m, 1))
                    break
                else:
                    samples.append((*zip(*self.env.parents(m)), 0, m, 0))
        p = self.env.cmols_to_batch(samples[-1][0])
        qp: FragmentGFlowNetOutput = self.sampling_model(p)
        qp.mol_preds = qp.mol_preds[:, 0]
        qsa_p = self.sampling_model.action_to_index(
            action=torch.tensor(samples[-1][1], device=self.env.device, dtype=torch.long),
            graph_batch=p,
            preds=qp
        )

        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        self.sampled_mols.append((r, m, trajectory_stats, inflow))
        # if self.replay_mode in ["online", "prioritized"]:
        #     m.reward = r
        #     self._add_mol_to_online(r, m, inflow)
        return samples
