import itertools
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


def flatten_iterable(iterable: Iterable) -> Iterable:
    return itertools.chain.from_iterable(iterable)


class BlockDataLoader(DataLoader):
    """Main `DataLoader` class which has been modified so as to read training data from disk in
    blocks, as opposed to a single line at a time (as is done in the original `DataLoader` class).

    From: https://github.com/MolecularAI/GraphINVENT/
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        block_size: int = 10000,
        shuffle: bool = True,
        n_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:

        # define variables to be used throughout dataloading
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.block_dataset = BlockDataset(
            self.dataset, batch_size=self.batch_size, block_size=self.block_size
        )

    def __iter__(self) -> torch.Tensor:

        # define a regular `DataLoader` using the `BlockDataset`
        block_loader = DataLoader(
            self.block_dataset, shuffle=self.shuffle, num_workers=self.n_workers
        )

        # define a condition for determining whether to drop the last block this
        # is done if the remainder block is very small (less than a tenth the
        # size of a normal block)
        condition = bool(
            int(self.block_dataset.__len__() / self.block_size)
            > 1 & self.block_dataset.__len__() % self.block_size
            < self.block_size / 10
        )

        # loop through and load BLOCKS of data every iteration
        for block in block_loader:
            block = [torch.squeeze(b) for b in block]

            # wrap each block in a `ShuffleBlock` so that data can be shuffled
            # within blocks
            batch_loader = DataLoader(
                dataset=ShuffleBlockWrapper(block),
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                pin_memory=self.pin_memory,
                drop_last=condition,
            )

            for batch in batch_loader:
                yield batch

    def __len__(self) -> int:
        # returns the number of graphs in the DataLoader
        n_blocks = len(self.dataset) // self.block_size
        n_rem = len(self.dataset) % self.block_size
        n_batch_per_block = self.__ceil__(self.block_size, self.batch_size)
        n_last = self.__ceil__(n_rem, self.batch_size)
        return n_batch_per_block * n_blocks + n_last

    def __ceil__(self, i: int, j: int) -> int:
        return (i + j - 1) // j


class BlockDataset(Dataset):
    """Modified `Dataset` class which returns BLOCKS of data when `__getitem__()` is called."""

    def __init__(self, dataset: Dataset, batch_size: int = 100, block_size: int = 10000) -> None:

        assert block_size >= batch_size, "Block size should be > batch size."

        self.block_size = block_size
        self.batch_size = batch_size
        self.dataset = dataset

    def __getitem__(self, idx: int) -> torch.Tensor:
        # returns a block of data from the dataset
        start = idx * self.block_size
        end = min((idx + 1) * self.block_size, len(self.dataset))
        return self.dataset[start:end]

    def __len__(self) -> int:
        # returns the number of blocks in the dataset
        return (len(self.dataset) + self.block_size - 1) // self.block_size


class ShuffleBlockWrapper:
    """Wrapper class used to wrap a block of data, enabling data to get shuffled.

    *within* a block.
    """

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __getitem__(self, idx: int) -> torch.Tensor:
        return [d[idx] for d in self.data]

    def __len__(self) -> int:
        return len(self.data[0])
