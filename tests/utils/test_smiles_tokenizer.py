import pytest

from rxitect.utils.smiles import SmilesTokenizer


@pytest.fixture(scope="module")
def example_smiles_list():
    return ["CCOc1ccc2[nH]nc(-c3cc(N4CC(F)C(F)C4)ncn3)c2c1",
            "N=C(N)Nc1nnn(Cc2cccc(Cn3nnc(C(=N)N)n3)c2)n1",
            "CCCCNC(=O)c1cnc(S)n1C(CC)c1ccc(F)c(F)c1",
            "Cc1cn(CC(=O)NC2CCN(c3c(F)cc4c(=O)c(C(=O)O)cn(C5CC5)c4c3Cl)C2)cn1",
            "O=c1cc(-c2ccc(Br)cc2)oc2c1ccc1ccccc12"]


def test_that_fit_results_in_vocabulary_with_unique_tokens(example_smiles_list):
    tokenizer = SmilesTokenizer()
    tokenizer.fit(example_smiles_list)
    
    assert len(set(tokenizer.vocabulary)) == len(tokenizer.vocabulary)


def test_that_encode_results_in_properly_shaped_tensor(example_smiles_list):
    tokenizer = SmilesTokenizer()
    tokenizer.fit(example_smiles_list)

    smiles_tensor = tokenizer.encode(example_smiles_list[0])
    assert smiles_tensor.shape == (1, 128)
