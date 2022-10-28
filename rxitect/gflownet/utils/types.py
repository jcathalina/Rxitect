from typing import NewType

import torch

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType('FlatRewards', torch.Tensor)

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType('RewardScalar', torch.Tensor)

# TODO: implement following new types/aliases: Trajectory, and ActionIndex
