import pytest
import torch
import torch.nn as nn
from pyprojroot import here
from torch.utils.data import DataLoader

from rxitect.data import SelfiesDataset, SmilesDataset
from rxitect.models import LSTMGenerator
from rxitect.tokenizers import SelfiesTokenizer, SmilesTokenizer, get_tokenizer


@pytest.fixture()
def smiles_tokenizer() -> SmilesTokenizer:
    test_vocabulary_filepath = here() / "tests/data/test_smiles_voc.txt"
    smiles_tokenizer = get_tokenizer("smiles", test_vocabulary_filepath, 100)
    return smiles_tokenizer


@pytest.fixture()
def smiles_dataloader(smiles_tokenizer: SmilesTokenizer) -> DataLoader:
    test_dataset_filepath = here() / "tests/data/test.smi"
    dataset = SmilesDataset(
        dataset_filepath=test_dataset_filepath, tokenizer=smiles_tokenizer
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=SmilesDataset.collate_fn,
    )
    return dataloader


def test_dataloader_loads_dataset_in_properly(smiles_dataloader: DataLoader):
    dataloader = smiles_dataloader
    assert len(dataloader.dataset) == 500
