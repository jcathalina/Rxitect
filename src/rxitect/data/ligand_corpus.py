from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data.dataloader import DataLoader
from globals import root_path

from rxitect.structs.vocabulary import Vocabulary
from rxitect.tensor_utils import random_split_frac


class LigandCorpus(pl.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        data_dir: Path = root_path / "data/processed",
        use_smiles: bool = False,
        batch_size: int = 512,
        n_workers: int = 1,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.data_dir = data_dir
        self.use_smiles = use_smiles
        self.batch_size = batch_size
        self.n_workers = n_workers

    def setup(self, stage: Optional[str] = None):
        corpus_filename = "smiles_ligand_corpus.txt" if self.use_smiles else "selfies_ligand_corpus.csv"
        ligand_full = pd.read_csv(self.data_dir / corpus_filename)["token"]

        if stage == "test" or stage is None:
            ligand_test = ligand_full.sample(frac=0.2, random_state=42)
            ligand_full = ligand_full.drop(
                ligand_test.index
            )  # Make sure the test set is excluded
            self.ligand_test = self.vocabulary.encode(
                [seq.split(" ") for seq in ligand_test]
            )

        if stage == "fit" or stage is None:
            ligand_train, ligand_val = random_split_frac(dataset=ligand_full)
            self.ligand_train = self.vocabulary.encode(
                [seq.split(" ") for seq in ligand_train]
            )
            self.ligand_val = self.vocabulary.encode(
                [seq.split(" ") for seq in ligand_val]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.ligand_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.n_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.ligand_val,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.n_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.ligand_test,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.n_workers,
        )
