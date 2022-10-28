from abc import ABC, abstractmethod
from typing import Union, Dict, List, Tuple

import torch
from rdkit.Chem import Mol

from rxitect.gflownet.utils.multiproc import MPModelPlaceholder
from rxitect.gflownet.utils.types import FlatRewards, RewardScalar


class IGraphTask(ABC):
    @abstractmethod
    def flat_reward_transform(self, y: Union[float, torch.Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    @abstractmethod
    def _load_task_models(self) -> Dict[str, MPModelPlaceholder]:
        pass

    @abstractmethod
    def sample_conditional_information(self, n: int) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def encode_conditional_information(self, info: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def compute_flat_rewards(self, mols: List[Mol]) -> Tuple[FlatRewards, torch.Tensor]:
        pass

    @staticmethod
    def cond_info_to_reward(cond_info: Dict[str, torch.Tensor], flat_rewards: FlatRewards) -> RewardScalar:
        if isinstance(flat_rewards, list):
            if isinstance(flat_rewards[0], torch.Tensor):
                flat_rewards = torch.stack(flat_rewards)
            else:
                flat_rewards = torch.tensor(flat_rewards)
        scalar_reward = (flat_rewards * cond_info['preferences']).sum(1)
        return scalar_reward**cond_info['beta']
