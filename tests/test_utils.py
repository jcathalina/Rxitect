import pytest

from rxitect import utils


def test_that_is_valid_smiles_works_as_expected():
    valid_smiles = "O=C(Nc1cccc(F)c1)N1CCN(c2ccnc(Cl)n2)CC1"
    invalid_smiles = "O=C(Nc1cccc(F)c1)N1CCN(c2cDnc(Cl)n2)CC1"
    assert utils.is_valid_smiles(valid_smiles)
    assert not utils.is_valid_smiles(invalid_smiles)
