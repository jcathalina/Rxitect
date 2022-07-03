import pandas as pd
import torch
from torch.utils.data import Dataset

from rxitect.utils.smiles import SmilesTokenizer


class SmilesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: SmilesTokenizer) -> None:
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        smiles = self.data["smiles"].iloc[index]
        tensor = self.tokenizer.encode(smiles)
        return tensor
