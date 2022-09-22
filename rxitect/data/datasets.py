import torch
from torch.utils.data import Dataset

from rxitect.tokenizers import SelfiesTokenizer, SmilesTokenizer


class SmilesDataset(Dataset):
    def __init__(self, dataset_filepath: str, tokenizer: SmilesTokenizer) -> None:
        self.tokenizer = tokenizer
        with open(dataset_filepath, "r") as f:
            self.smiles = [line.split()[0] for line in f]

    def __getitem__(self, index: int) -> torch.Tensor:
        smiles = self.smiles[index]
        return self.tokenizer.encode(smiles)

    def __len__(self):
        return len(self.smiles)

    def __str__(self) -> str:
        return f"SMILES Dataset containing {len(self)} structures"

    @classmethod
    def collate_fn(cls, arr: torch.Tensor) -> torch.Tensor:
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_len = max([seq.size(0) for seq in arr])
        collated_arr = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, seq in enumerate(arr):
            collated_arr[i, : seq.size(0)] = seq
        return collated_arr


class SelfiesDataset(Dataset):
    def __init__(self, dataset_filepath: str, tokenizer: SelfiesTokenizer) -> None:
        self.tokenizer = tokenizer
        with open(dataset_filepath, "r") as f:
            self.selfies = [line.split()[0] for line in f]

    def __getitem__(self, index: int) -> torch.Tensor:
        selfies = self.selfies[index]
        return self.tokenizer.encode(selfies)

    def __len__(self):
        return len(self.selfies)

    def __str__(self) -> str:
        return f"SELFIES Dataset containing {len(self)} structures"

    @classmethod
    def collate_fn(cls, arr: torch.Tensor) -> torch.Tensor:
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_len = max([seq.size(0) for seq in arr])
        collated_arr = torch.zeros(len(arr), max_len)
        for i, seq in enumerate(arr):
            collated_arr[i, : seq.size(0)] = seq
        return collated_arr
