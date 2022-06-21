from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import selfies as sf
import torch
import pytorch_lightning as pl
from tqdm import trange

from rxitect.chem.utils import randomize_selfies
from rxitect.utils.types import PathLike


class SelfiesQsarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pad_to_len: int = 128, augment: bool = True, train: bool = True):
        super().__init__()
        self.pad_to_len = pad_to_len
        if train:
            df = df[df["document year"] <= 2015]
        else:
            df = df[df["document year"] > 2015]
            
        if augment:
            aug_df = df.copy()
            aug_df['selfies'] = [randomize_selfies(selfies) for selfies in aug_df['selfies'].values]
            df = pd.concat([df, aug_df])

        df = df[df['selfies'].apply(sf.len_selfies) <= self.pad_to_len]
        self.vocab = sf.get_alphabet_from_selfies(df['selfies'])
        stoi = {x:i for i,x in enumerate(self.vocab, start=2)}
        stoi['[nop]'] = 0
        stoi['.'] = 1

        inp = [sf.selfies_to_encoding(selfies=selfies, vocab_stoi=stoi, pad_to_len=self.pad_to_len, enc_type='label') for selfies in df['selfies']]
        self.length = torch.from_numpy(np.array([np.count_nonzero(enc_selfies) for enc_selfies in inp]))
        self.target = torch.from_numpy(df['pchembl value'].values)
        self.inp = torch.tensor(inp, dtype=torch.float32)

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, index):
        sample = {}
        sample['tokenized_smiles'] = (self.inp[index])
        sample['length'] = (self.length[index])
        sample['labels'] = (self.target[index])
        return sample


class SelfiesQsarDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        
    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.qsar_train = SelfiesQsarDataset(self.df, train=True)
        if stage in (None, "test", "predict"):
            self.qsar_test = SelfiesQsarDataset(self.df, train=False)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.qsar_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          sampler=None)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.qsar_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          sampler=None)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.qsar_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          sampler=None)
