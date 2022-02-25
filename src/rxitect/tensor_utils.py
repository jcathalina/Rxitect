from typing import Union
from torch.utils.data.dataset import Dataset, random_split

import numpy as np
import torch


def unique(arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Finds unique rows in arr and return their indices

    Args:
        arr: a tensor containing all generated encoded SMILES
    Returns:

    """
    if type(arr) == torch.Tensor:
        arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    )
    _, idxs = np.unique(arr_, return_index=True)
    idxs = np.sort(idxs)
    if type(arr) == torch.Tensor:
        idxs = torch.LongTensor(idxs).to(arr.get_device())
    return idxs


def random_split_frac(dataset: Dataset, train_frac: float = 0.9, val_frac: float = 0.1):
    """
    Helper wrapper function around PyTorch's random_split method that allows you to pass
    fractions instead of integers.
    """
    if train_frac + val_frac != 1:
        raise ValueError("The fractions have to add up to 1.")

    dataset_size = len(dataset)

    len_1 = int(np.floor(train_frac * dataset_size))
    len_2 = dataset_size - len_1
    return random_split(dataset=dataset, lengths=[len_1, len_2])


class ClippedScore:
    r"""
    Clips a score between specified low and high scores, and does a linear interpolation in between.
    The function looks like this:
       upper_x < lower_x                 lower_x < upper_x
    __________                                   ____________
              \                                 /
               \                               /
                \__________          _________/
    This class works as follows:
    First the input is mapped onto a linear interpolation between both specified points.
    Then the generated values are clipped between low and high scores.
    """

    def __init__(
        self, upper_x: float, lower_x=0.0, high_score=1.0, low_score=0.0
    ) -> None:
        """
        Args:
            upper_x: x-value from which (or until which if smaller than lower_x) the score is maximal
            lower_x: x-value until which (or from which if larger than upper_x) the score is minimal
            high_score: maximal score to clip to
            low_score: minimal score to clip to
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        self.slope = (high_score - low_score) / (upper_x - lower_x)
        self.intercept = high_score - self.slope * upper_x

    def __call__(self, x):
        y = self.slope * x + self.intercept
        return np.clip(y, self.low_score, self.high_score)


class SmoothClippedScore:
    """
    Smooth variant of ClippedScore.
    Implemented as a logistic function that has the same steepness as ClippedScore in the
    center of the logistic function.
    """

    def __init__(
        self, upper_x: float, lower_x=0.0, high_score=1.0, low_score=0.0
    ) -> None:
        """
        Args:
            upper_x: x-value from which (or until which if smaller than lower_x) the score approaches high_score
            lower_x: x-value until which (or from which if larger than upper_x) the score approaches low_score
            high_score: maximal score (reached at +/- infinity)
            low_score: minimal score (reached at -/+ infinity)
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        # Slope of a standard logistic function in the middle is 0.25 -> rescale k accordingly
        self.k = 4.0 / (upper_x - lower_x)
        self.middle_x = (upper_x + lower_x) / 2
        self.L = high_score - low_score

    def __call__(self, x):
        return self.low_score + self.L / (1 + np.exp(-self.k * (x - self.middle_x)))