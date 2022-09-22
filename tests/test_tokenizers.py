import pytest
from pyprojroot import here

from rxitect.tokenizers import SelfiesTokenizer, SmilesTokenizer, get_tokenizer


@pytest.fixture
def smiles_tokenizer() -> SmilesTokenizer:
    test_vocabulary_filepath = here() / "tests/data/test_smiles_voc.txt"
    smiles_tokenizer = get_tokenizer("smiles", test_vocabulary_filepath, 100)
    return smiles_tokenizer


@pytest.fixture
def selfies_tokenizer() -> SelfiesTokenizer:
    test_vocabulary_filepath = here() / "tests/data/test_selfies_voc.txt"
    smiles_tokenizer = get_tokenizer("selfies", test_vocabulary_filepath, 100)
    return smiles_tokenizer


def test_decoding_encoded_smiles_reconstructs_smiles_correctly(smiles_tokenizer):
    sample_smiles = "CCBr[nH]"
    tokenizer = smiles_tokenizer
    encoded_smiles = tokenizer.encode(sample_smiles)
    decoded_smiles = tokenizer.decode(encoded_smiles)

    assert decoded_smiles == sample_smiles


def test_decoding_encoded_selfies_reconstructs_selfies_correctly(selfies_tokenizer):
    sample_selfies = "[C][C][Br][NH1]"
    tokenizer = selfies_tokenizer
    encoded_selfies = tokenizer.encode(sample_selfies)
    decoded_smiles = tokenizer.decode(encoded_selfies)

    assert decoded_smiles == sample_selfies
