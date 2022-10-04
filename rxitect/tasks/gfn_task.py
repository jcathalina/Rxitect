from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from rdkit.Chem.rdchem import Mol
from torch import nn

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
ScalarReward = NewType("ScalarReward", Tensor)  # type: ignore


class GFNTask(ABC):
    @abstractmethod
    def flat_reward_transform(self, y: Union[float, torch.Tensor]) -> FlatRewards:
        pass

    @abstractmethod
    def inverse_flat_reward_transform(
        self, rp: FlatRewards
    ) -> Union[float, torch.Tensor]:
        pass

    @abstractmethod
    def _load_task_models(self) -> Dict[str, nn.Module]:
        pass

    @abstractmethod
    def sample_conditional_information(self, n: int) -> Dict[str, Any]:
        """
        Parameters
        ----------
        n: size of random sample

        Returns
        -------
        Dictionary containing conditional information
        """
        pass

    @abstractmethod
    def cond_info_to_reward(
        self, cond_info: Dict[str, torch.Tensor], flat_rewards: FlatRewards
    ) -> ScalarReward:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.
        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional information (e.g. temperature)
        flat_rewards: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.
        Returns
        -------
        reward: ScalarReward
            A 1d tensor, a scalar reward for each minibatch entry.
        """
        pass

    @abstractmethod
    def compute_flat_rewards(self, mols: List[Mol]) -> Tuple[FlatRewards, torch.Tensor]:
        """Compute the flat rewards of mols according the tasks' proxies
        Parameters
        ----------
        mols: List[Mol]
            A list of RDKit molecules.
        Returns
        -------
        reward: FlatRewards
            A 2d tensor, a vector of scalar reward for valid each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        pass
