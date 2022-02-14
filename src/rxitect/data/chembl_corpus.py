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


class ChemblCorpus(pl.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        data_dir: Path = root_path / "data/processed",
        use_smiles: bool = False,
        batch_size: int = 512,
        n_workers: int = 1,
        dev_run: bool = False,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.data_dir = data_dir
        self.use_smiles = use_smiles
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dev_run = dev_run

        self.chembl_train = None
        self.chembl_test = None
        self.chembl_val = None

    def setup(self, stage: Optional[str] = None):
        corpus_filename = "smiles_chembl_corpus.txt" if self.use_smiles else "selfies_chembl_corpus.csv"

        chembl_full = pd.read_csv(self.data_dir / corpus_filename, nrows=100_000 if self.dev_run else 200_000)["token"]

        if stage == "test" or stage is None:
            chembl_test = chembl_full.sample(frac=0.2, random_state=42)
            chembl_full = chembl_full.drop(
                chembl_test.index
            )  # Make sure the test set is excluded
            self.chembl_test = self.vocabulary.encode(
                [seq.split(" ") for seq in chembl_test]
            )

        if stage == "fit" or stage is None:
            chembl_train, chembl_val = random_split_frac(dataset=chembl_full)
            self.chembl_train = self.vocabulary.encode(
                [seq.split(" ") for seq in chembl_train]
            )
            self.chembl_val = self.vocabulary.encode(
                [seq.split(" ") for seq in chembl_val]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.chembl_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
            num_workers=self.n_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.chembl_val,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=False,
            num_workers=self.n_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.chembl_test,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=False,
            num_workers=self.n_workers,
        )
