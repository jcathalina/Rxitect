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
        smiles = self.data[self.vocabulary.mol_str_type].iloc[index]
        tensor = self.vocabulary.encode(smiles)
        return tensor
