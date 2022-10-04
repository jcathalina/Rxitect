from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch

from rxitect.envs.contexts.graph_env_context import (
    StateActionPair, generate_forward_trajectory)

if TYPE_CHECKING:
    from rxitect.envs.contexts import (Action, ActionCategorical, ActionIndex,
                                       Graph)


class SamplingModel(nn.Module):
    def forward(self, batch: Batch) -> Tuple[ActionCategorical, torch.Tensor]:
        raise NotImplementedError()

    def log_z(self, cond_info: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class GFNAlgorithm(ABC):
    @abstractmethod
    def create_training_data_from_own_samples(
        self, model: SamplingModel, n: int, cond_info: torch.Tensor
    ) -> List[Dict]:
        """Generate trajectories by sampling a model
        Parameters
        ----------
        model: SamplingModel
           The model being sampled
        n: int
            Number of samples
        cond_info: torch.Tensor
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
        pass

    @staticmethod
    def create_training_data_from_graphs(graphs: List[Graph]) -> List[Trajectory]:
        """Generate trajectories from known endpoints
        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{"traj": generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, rewards) -> Batch:
        """Construct a batch from a list of trajectories and their information
        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        rewards: Tensor
            The transformed reward (e.g. R(x) ** beta) for each trajectory. Shape (N,)
        Returns
        -------
        batch: Batch
             A (CPU) Batch object with relevant attributes added
        """
        pass

    @abstractmethod
    def compute_batch_losses(
        self, model: nn.Module, batch: Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the loss for a batch of data, and proves logging information
        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).
        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        pass


Trajectory = Dict[str, List[StateActionPair]]
