import logging
from pathlib import Path
import re
from typing import List
import pandas as pd

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from rxitect.utils.common import flatten_iterable
from rxitect.utils.tokenizer import Tokenizer

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(name=__name__)


class SmilesTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, mol_strings: List[str]) -> None:
        """"""
        unique_tokens = set(
            flatten_iterable([self.tokenize(mol_string) for mol_string in mol_strings])
        )
        self.vocabulary = self.sentinel_tokens + list(unique_tokens)

        assert self.vocabulary[0] == self.pad_token, "Pad token has to be at index '0'"

        self.num_tokens = len(self.vocabulary)
        self.stoi = dict(zip(self.vocabulary, range(self.num_tokens)))
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def tokenize(cls, mol_string: str) -> List[str]:
        """Method that takes the string representation of a molecule (in this case, SMILES) and
        returns a list of tokens that the string is made up of.

        Args:
            mol_string (str): The SMILES representation of a molecule
        Returns:
            A list of tokens that make up the passed SMILES.
        """
        tokens = []
        regex = "(\[[^\[\]]{1,6}\])"
        mol_string = re.sub("\[\d+", "[", mol_string)
        mol_string = mol_string.replace("Br", "R").replace("Cl", "L")
        for word in re.split(regex, mol_string):
            if word == "" or word is None:
                continue
            if word.startswith("["):
                tokens.append(word)
            else:
                for char in word:
                    tokens.append(char)
        return tokens

    @classmethod
    def detokenize(cls, tokenized_mol_string: List[str]) -> str:
        """Takes an array of indices and returns the corresponding SMILES."""
        smiles = "".join(tokenized_mol_string)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def encode(self, mol_string: str) -> torch.Tensor:
        """Takes a list containing tokens from the passed SMILES (eg '[NH]') and encodes to array
        of indices."""
        tokenized_mol_string = [self.start_token] + self.tokenize(mol_string) + [self.end_token]
        encoded_smiles = torch.zeros(128, dtype=torch.long)
        for index, token in enumerate(tokenized_mol_string):
            encoded_smiles[index] = self.stoi.get(token, self.stoi[self.missing_token])
        return encoded_smiles

    def batch_encode(self, mol_strings: List[str]) -> torch.Tensor:
        tokenized_mol_strings = [
            [self.start_token] + self.tokenize(mol_string) + [self.end_token]
            for mol_string in mol_strings
        ]
        encoded_smiles = torch.zeros(size=(len(tokenized_mol_strings), 128), dtype=torch.long)
        for i, tokenized_smiles in enumerate(tokenized_mol_strings):
            for j, token in enumerate(tokenized_smiles):
                encoded_smiles[i, j] = self.stoi.get(token, self.stoi[self.missing_token])
        return encoded_smiles

    def decode(self, mol_tensor: torch.Tensor) -> str:
        """Takes an array of indices and returns the corresponding SMILES."""
        tokens = [self.itos[index] for index in mol_tensor]
        smiles = self.detokenize(tokens)
        smiles = smiles.replace(self.start_token, "")
        smiles = smiles.replace(self.end_token, "")
        return smiles

    def fit_from_file(self, vocabulary_filepath: Path) -> None:
        tokens = pd.read_csv(vocabulary_filepath, header=None).values.flatten()
        unique_tokens = set([t for t in tokens if t not in self.sentinel_tokens])
        self.vocabulary = self.sentinel_tokens + list(unique_tokens)

        assert self.vocabulary[0] == self.pad_token, "Pad token has to be at index '0'"

        self.num_tokens = len(self.vocabulary)
        self.stoi = dict(zip(self.vocabulary, range(self.num_tokens)))
        self.itos = {index: token for token, index in self.stoi.items()}


def clean_and_canonalize(smiles: str) -> str:
    """Removes charges and canonalizes the SMILES representation of a molecule.

    Args:
        smiles (str): SMILES string representation of a molecule.
    Returns:
        Cleaned (uncharged version of largest fragment) & Canonicalized SMILES,
        or empty string on invalid Mol.
    """

    processed_smiles = ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = rdMolStandardize.ChargeParent(mol)
        smiles = Chem.MolToSmiles(smiles)
        processed_smiles = Chem.CanonSmiles(smiles)
    except Exception as e:
        print("SMILES Parsing Error: ", e)
    return processed_smiles


def randomize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )
