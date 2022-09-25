import logging
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, random_split

from rxitect.data import SmilesDataset
from rxitect.tokenizers import SmilesTokenizer

logger = logging.getLogger(__name__)


class SmilesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_filepath: str,
        tokenizer=SmilesTokenizer,
        train_val_test_split: Tuple[int, int, int] = (1_500_000, 185_000, 186_227),
        batch_size: int = 128,
        num_workers: int = 0,
        num_partitions: Optional[int] = None,
        pin_memory: bool = False,
        random_state: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.dataset_filepath = dataset_filepath
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_partitions = num_partitions
        self.pin_memory = pin_memory
        self.random_state = random_state
        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        pass
        # TODO: Download the tokenized ChEMBL file here
        # saves us the params if we init the vocab internally as well.

    def setup(self, stage: Optional[str] = None) -> None:
        # TODO: Make ChEMBL v30 a downloadable dataset like MNIST from torch and simplify
        data = SmilesDataset(self.dataset_filepath, self.tokenizer)
        # Create splits for train/val/test
        self.train_data, self.val_data, self.test_data = random_split(
            dataset=data,
            lengths=self.train_val_test_split,
            generator=torch.Generator().manual_seed(self.random_state),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=SmilesDataset.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=SmilesDataset.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=SmilesDataset.collate_fn,
            shuffle=False,
        )
