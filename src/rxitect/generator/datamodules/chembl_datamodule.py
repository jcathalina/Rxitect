import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from rxitect.generator.datamodules.components.molecule_dataset import MoleculeDataset
from pytorch_lightning import LightningDataModule

from rxitect.structs.vocabulary import SelfiesVocabulary, SmilesVocabulary
from rdkit import Chem
import pandas as pd
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

tqdm.pandas()
logger = logging.getLogger(__name__)


class ChemblSmilesDataModule(LightningDataModule):
    def __init__(self,
                 chembl_smiles_filepath: Path,
                 train_val_test_split: Tuple[int, int, int] = (70_000, 10_000, 20_000),
                 augment: bool = True,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 pin_memory: bool = False,) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.chembl_smiles_filepath = chembl_smiles_filepath
        self.vocabulary = SmilesVocabulary()
    
    def prepare_data(self) -> None:
        pass
        # TODO: Download the tokenized ChEMBL file here, saves us the params if we init the vocab internally as well.
        
    def augment_smiles(self, smiles: str) -> str:
        np.random.seed(42)  # TODO: pass seed number from config?

        mol = Chem.MolFromSmiles(smiles)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol,atom_idxs)
        return Chem.MolToSmiles(mol, canonical=False)

    def setup(self, stage: Optional[str] = None) -> None:
        num_samples = sum(self.hparams.train_val_test_split)
        data = pd.read_table(self.chembl_smiles_filepath, usecols=["smiles"]).sample(n=num_samples, random_state=42)

        if self.hparams.augment:
            #Atom order randomize SMILES
            logger.info("Randomizing SMILES...")
            data["smiles"] = data["smiles"].progress_apply(self.augment_smiles)
        
        #Initialize Vocabulary
        self.vocabulary.fit(data.smiles.values)

        data = MoleculeDataset(data, self.vocabulary)
        #Create splits for train/val
        self.train_data, self.val_data, self.test_data = random_split(
                dataset=data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
        )
    
    def custom_collate_and_pad(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            batch (List[str]): A list of vectorized smiles.

        Returns:
            A list containing the padded versions of the tensors that were passed in.
        """
        tensors = [torch.tensor(vectorized_smiles) for vectorized_smiles in batch]
        #pad and transpose, pytorch RNNs  (and now transformers) expect (sequence,batch, features) batch) dimensions
        tensors = torch.nn.utils.rnn.pad_sequence(tensors)
        return tensors

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


class ChemblSelfiesDataModule(LightningDataModule):
    def __init__(self, chembl_smiles_filepath: Path, vocabulary: SelfiesVocabulary):
        super().__init__()
        self.chembl_smiles_filepath = chembl_smiles_filepath
        self.vocabulary = vocabulary
        self.val_size = 200
        self.train_size = 20_000
        self.batch_size = 128
    
    def prepare_data(self) -> None:
        pass
        # TODO: Download the tokenized ChEMBL file here, saves us the params if we init the vocab internally as well.

    def augment_selfies(self, selfies: str) -> str:
        pass
        # TODO take from chem utils

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = dd.read_table(self.chembl_smiles_filepath).head(n=20_000)
        #Create splits for train/val
        np.random.seed(seed=42)
        idxs = np.array(range(len(self.data)))
        np.random.shuffle(idxs)
        val_idxs, train_idxs = idxs[:self.val_size], idxs[self.val_size:self.val_size+self.train_size]
        self.train_data = self.data.iloc[train_idxs]
        self.val_data = self.data.iloc[val_idxs]

        # TODO: initialize Vocabulary in here maybe to make things easier...
        # TODO: Add augmentation by randomization here.... (relies on the above step)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = MoleculeDataset(self.train_data, self.vocabulary)
        return DataLoader(dataset=dataset,
                         batch_size=self.batch_size,
                         pin_memory=True,
                         shuffle=True,)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = MoleculeDataset(self.val_data, self.vocabulary)
        return DataLoader(dataset=dataset,
                         batch_size=self.batch_size,
                         pin_memory=True,
                         shuffle=False,)
