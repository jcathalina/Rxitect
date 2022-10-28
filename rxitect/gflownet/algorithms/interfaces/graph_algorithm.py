from abc import ABC, abstractmethod
from typing import List

from networkx import Graph
from torch import Tensor
import torch.nn as nn
import torch_geometric.data as gd
from torch_geometric.data import Batch

from rxitect.gflownet.utils.graph import generate_forward_trajectory


class IGraphAlgorithm(ABC):
    illegal_action_logreward: float
    bootstrap_own_reward: bool

    @abstractmethod
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
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        pass

    @staticmethod
    def create_training_data_from_graphs(graphs: List[Graph]):
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
        return [{'traj': generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, rewards):
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
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        pass

    def compute_batch_losses(self, model: nn.Module, batch: Batch, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
            A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
            Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: Batch
            Batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
            the number of trajectories for which the reward loss is computed. Ignored if 0."""
        pass
