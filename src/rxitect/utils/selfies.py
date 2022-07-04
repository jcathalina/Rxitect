import logging
from pathlib import Path
from typing import List

import pandas as pd
import selfies as sf
import torch
from rdkit import Chem, RDLogger

from rxitect.utils.tokenizer import Tokenizer

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(name=__name__)


class SelfiesTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pad_token = "[nop]"
        self.sentinel_tokens[0] = self.pad_token
        self.max_len = 110  # Max tokens found in SELFIES corpus.

    def fit(self, mol_strings: List[str]) -> None:
        """"""
        unique_tokens = sf.get_alphabet_from_selfies(mol_strings)
        self.vocabulary = self.sentinel_tokens + list(unique_tokens)

        print(self.vocabulary)
        assert self.vocabulary[0] == self.pad_token, "Pad token has to be at index '0'"

        self.num_tokens = len(self.vocabulary)
        self.stoi = dict(zip(self.vocabulary, range(self.num_tokens)))
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def tokenize(cls, mol_string: str) -> List[str]:
        """Method that takes the string representation of a molecule (in this case, SELFIES) and
        returns a list of tokens that the string is made up of.

        Args:
            mol_string (str): The SELFIES representation of a molecule
        Returns:
            A list of tokens that make up the passed SELFIES.
        """
        tokens = list(sf.split_selfies(mol_string))
        return tokens

    @classmethod
    def detokenize(cls, tokenized_mol_string: List[str]) -> str:
        """Takes an array of indices and returns the corresponding SELFIES."""
        selfies = "".join(tokenized_mol_string)
        return selfies

    def encode(self, mol_string: str) -> torch.Tensor:
        """Takes a list containing tokens from the passed SELFIES (eg '[NH]') and encodes to array
        of indices."""
        tokenized_mol_string = self.tokenize(mol_string)
        encoded_selfies = torch.zeros(self.max_len, dtype=torch.long)
        for index, token in enumerate(tokenized_mol_string):
            encoded_selfies[index] = self.stoi.get(token, self.stoi[self.missing_token])
        return encoded_selfies

    def batch_encode(self, mol_strings: List[str]) -> torch.Tensor:
        """"""
        tokenized_mol_strings = [self.tokenize(mol_string) for mol_string in mol_strings]
        encoded_selfies = torch.zeros(len(tokenized_mol_strings), self.max_len, dtype=torch.long)
        for i, tokenized_selfies in enumerate(tokenized_mol_strings):
            for j, token in enumerate(tokenized_selfies):
                encoded_selfies[i, j] = self.stoi.get(token, self.stoi[self.missing_token])
        return encoded_selfies

    def decode(self, mol_tensor: torch.Tensor) -> str:
        """Takes an array of indices and returns the corresponding SELFIES."""
        tokens = []
        for i in mol_tensor:
            token = self.itos[i.item()]
            if token == self.end_token:
                break
            if token in self.sentinel_tokens:
                continue
            tokens.append(token)
        mol_str = "".join(tokens)
        return mol_str

    def fit_from_file(self, vocabulary_filepath: Path) -> None:
        tokens = pd.read_csv(vocabulary_filepath, header=None).values.flatten()
        unique_tokens = set([t for t in tokens if t not in self.sentinel_tokens])
        self.vocabulary = self.sentinel_tokens + list(unique_tokens)

        assert self.vocabulary[0] == self.pad_token, "Pad token has to be at index '0'"

        self.num_tokens = len(self.vocabulary)
        self.stoi = dict(zip(self.vocabulary, range(self.num_tokens)))
        self.itos = {index: token for token, index in self.stoi.items()}


def randomize_selfies(selfies: str) -> str:
    smiles = sf.decoder(selfies)
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    randomized_smiles = Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )
    return sf.encoder(randomized_smiles)
