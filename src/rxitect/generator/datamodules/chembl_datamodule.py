from pathlib import Path
from typing import Optional
from rxitect.generator.datamodules.components.molecule_dataset import MoleculeDataset
from pytorch_lightning import LightningDataModule

from rxitect.structs.vocabulary import SelfiesVocabulary, SmilesVocabulary, Vocabulary
from rdkit import Chem
import dask.dataframe as dd
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader


class ChemblSmilesDataModule(LightningDataModule):
    def __init__(self, chembl_smiles_filepath: Path, vocabulary: SmilesVocabulary):
        super().__init__()
        self.chembl_smiles_filepath = chembl_smiles_filepath
        self.vocabulary = vocabulary
        self.val_size = 100
        self.train_size = 10_000
        self.batch_size = 128
    
    def prepare_data(self) -> None:
        pass
        # TODO: Download the tokenized ChEMBL file here, saves us the params if we init the vocab internally as well.
        

    def augment_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol,atom_idxs)
        return Chem.MolToSmiles(mol, canonical=False)

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
                         shuffle=True,)
