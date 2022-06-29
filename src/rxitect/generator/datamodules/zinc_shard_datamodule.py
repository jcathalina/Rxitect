import logging
from pathlib import Path
from typing import Optional, Tuple
import os

import torch
from rxitect.generator.datamodules.components.molecule_dataset import MoleculeDataset
from pytorch_lightning import LightningDataModule

from rxitect.structs.vocabulary import SelfiesVocabulary, SmilesVocabulary, Vocabulary
from rdkit import Chem
# import dask.dataframe as dd  # dask partition problem
import pandas as pd
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

tqdm.pandas()
logger = logging.getLogger(__name__)


class ZincShardDataModule(LightningDataModule):
    def __init__(self,
                 zinc_smiles_filepath: Path,
                 train_val_test_split: Tuple[int, int, int],
                 batch_size: int = 256,
                 num_workers: int = 4,
                 pin_memory: bool = False,) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.zinc_smiles_filepath = zinc_smiles_filepath
        self.vocabulary = SmilesVocabulary()
        self.shard = "EEBD"  # Shard must be > self.val_size + self.train_size

    def prepare_data(self):
        # called only on 1 GPU, no assignments of state!
        #e.g. use to download_dataset(), tokenize(), build_vocab()
        # os.system(f"mkdir -pv {self.shard[0:2]} && wget -c http://files.docking.org/2D/{self.shard[0:2]}/{self.shard}.smi -O {self.shard[0:2]}/{self.shard}.smi")
        pass

    def randomize_smiles_atom_order(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol,atom_idxs)
        return Chem.MolToSmiles(mol, canonical=False)
    
    def custom_collate_and_pad(self, batch):
        #Batch is a list of vectorized smiles
        tensors = [torch.tensor(l) for l in batch]
        #pad and transpose, pytorch RNNs  (and now transformers) expect (sequence,batch, features) batch) dimensions
        tensors = torch.nn.utils.rnn.pad_sequence(tensors)
        return tensors
    
    def setup(self, stage: Optional[str] = None):
        #Load data
        data = pd.read_csv(self.zinc_smiles_filepath, sep = " ", nrows = sum(self.hparams.train_val_test_split))
        #Atom order randomize SMILES
        logger.info("Randomizing SMILES...")
        data["smiles"] = data["smiles"].progress_apply(self.randomize_smiles_atom_order)
         
        #Initialize Vocabulary
        self.vocabulary.fit(data.smiles.values)
        
        #Create splits for train/val
        data = MoleculeDataset(data, self.vocabulary)

        self.train_data, self.val_data, self.test_data = random_split(
                dataset=data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_data,
                         batch_size=self.hparams.batch_size,
                         pin_memory=self.hparams.pin_memory,
                         num_workers=self.hparams.num_workers,
                         collate_fn=self.custom_collate_and_pad,
                         shuffle=True,)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val_data,
                         batch_size=self.hparams.batch_size,
                         pin_memory=self.hparams.pin_memory,
                         num_workers=self.hparams.num_workers,
                         collate_fn=self.custom_collate_and_pad,
                         shuffle=False,)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.test_data,
                         batch_size=self.hparams.batch_size,
                         pin_memory=self.hparams.pin_memory,
                         num_workers=self.hparams.num_workers,
                         collate_fn=self.custom_collate_and_pad,
                         shuffle=False,)
