import torch
from torch.utils.data import Dataset

from rxitect.structs.vocabulary import Vocabulary

class MoleculeDataset(Dataset):
    def __init__(self, data, vocabulary: Vocabulary) -> None:
        self.data = data
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        token_seq = self.data.token.iloc[index]
        tensor = self.vocabulary.encode([token_seq.split(" ")]).reshape(-1)  # Reshape because encode takes list of lists.

        # TODO: Maybe rewrite this to tokenize & encode on-the-fly?
        return tensor