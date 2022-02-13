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
