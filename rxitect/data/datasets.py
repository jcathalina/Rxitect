import torch

from rxitect.tokenizers import SelfiesTokenizer, SmilesTokenizer

import tarfile

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(self, dataset_filepath: str, tokenizer: SmilesTokenizer) -> None:
        self.tokenizer = tokenizer
        self.padding_value = tokenizer.tk2ix_[tokenizer.pad_token]
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


class QM9Dataset(Dataset):
    def __init__(self, h5_file=None, xyz_file=None, train=True, target='gap', split_seed=142857, ratio=0.9):
        if h5_file is not None:
            self.df = pd.HDFStore(h5_file, 'r')['df']
        elif xyz_file is not None:
            self.load_tar()
        rng = np.random.default_rng(split_seed)
        idcs = np.arange(len(self.df))  # TODO: error if there is no h5_file provided. Should h5 be required
        rng.shuffle(idcs)
        self.target = target
        if train:
            self.idcs = idcs[:int(np.floor(ratio * len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(self.df))):]

    def get_stats(self, percentile=0.95):
        y = self.df[self.target]
        return y.min(), y.max(), np.sort(y)[int(y.shape[0] * percentile)]

    def load_tar(self, xyz_file):
        f = tarfile.TarFile(xyz_file, 'r')
        labels = ['rA', 'rB', 'rC', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        all_mols = []
        for pt in f:
            pt = f.extractfile(pt)
            data = pt.read().decode().splitlines()
            all_mols.append(data[-2].split()[:1] + list(map(float, data[1].split()[2:])))
        self.df = pd.DataFrame(all_mols, columns=['SMILES'] + labels)

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return Chem.MolFromSmiles(self.df['SMILES'][self.idcs[idx]]), self.df[self.target][self.idcs[idx]]
