import pandas as pd
import torch
from torch.utils.data import Dataset

from rxitect.utils.selfies import SelfiesTokenizer


class SelfiesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: SelfiesTokenizer) -> None:
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        selfies = self.data["selfies"].iloc[index]
        tensor = self.tokenizer.encode(selfies)
        return tensor
