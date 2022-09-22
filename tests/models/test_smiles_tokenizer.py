import pytest

from rxitect.tokenizers import get_tokenizer, SmilesTokenizer
from pyprojroot import here


@pytest.fixture
def smiles_tokenizer() -> SmilesTokenizer:
    test_vocabulary_filepath = here() / "tests/data/test_voc.txt"
    smiles_tokenizer = get_tokenizer("smiles", test_vocabulary_filepath, 100)
    return smiles_tokenizer


def test_decoding_encoded_smiles_reconstructs_smiles_correctly(smiles_tokenizer):
    sample_smiles = ["CCC", "CCBr[nH]"]
    tokenizer = smiles_tokenizer
    encoded_smiles = tokenizer.encode(sample_smiles)
    decoded_smiles = tokenizer.decode(encoded_smiles)
    
    assert decoded_smiles == sample_smiles
