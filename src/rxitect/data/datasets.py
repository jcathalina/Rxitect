from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from numpy.typing import ArrayLike

from rxitect.chem.utils import calc_fp, mol_from_selfies
from rxitect.data.vectorizer import SelfiesVectorizer
from rxitect.utils.types import ArrayDict


from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
import dask.dataframe as dd
import selfies as sf
import torch
import pytorch_lightning as pl


class SelfiesQsarDataset(Dataset):
    def __init__(self, smiles: List[str], pchembl_values: List[float], vectorizer: SelfiesVectorizer, pad_to_len: int = 128):
        super().__init__()
        self.smiles = smiles
        self.selfies = [sf.encoder(smi) for smi in self.smiles]
        self.target = pchembl_values
        self.vectorizer = vectorizer
        
        self.vectorizer.pad = pad_to_len
        self.vectorizer.fit(selfies_list=self.selfies)
        # self.vocab = sf.get_alphabet_from_selfies(self.selfies)
        # stoi = {x:i for i, x in enumerate(vocab, start=2)}
        # stoi['[nop]'] = 0
        # stoi['.'] = 1
        
        # enc = [sf.selfies_to_encoding(selfies=selfies, vocab_stoi=stoi, pad_to_len=128, enc_type='label') for selfies in df['selfies']]
        # df['enc_selfies'] = enc
        # df = df[df['enc_selfies'].apply(len) <= 128]
        
        # self.inp = [sf.selfies_to_encoding(selfies=selfies, vocab_stoi=stoi, pad_to_len=128, enc_type='label') for selfies in df['selfies']]
        self.inp = [vectorizer.transform(s) for  s in self.selfies]
        # self.length = np.array([np.count_nonzero(enc_selfies) for enc_selfies in self.inp])
        # self.target = df['pchembl_value'].values
        

        self.inp = torch.tensor(self.inp, dtype=torch.float32)
        self.length = torch.from_numpy(self.length)
        self.target = torch.from_numpy(self.target)

    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, index):
        sample = {}
        sample['tokenized_smiles'] = (self.inp[index])
        sample['length'] = (self.length[index])
        sample['labels'] = (self.target[index])
        return sample


class SelfiesQsarDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128, max_selfies_len: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.max_selfies_len = max_selfies_len
        
    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.qsar_train = SelfiesQsarDataset(train=True)
        if stage in (None, "test", "predict"):
            self.qsar_test = SelfiesQsarDataset(train=False)
    
    def train_dataloader(self):
        return DataLoader(self.qsar_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          sampler=None)

    def test_dataloader(self):
        return DataLoader(self.qsar_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          sampler=None)
    
    def predict_dataloader(self):
        return DataLoader(self.qsar_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          sampler=None)