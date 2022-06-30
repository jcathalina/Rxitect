from asyncio.log import logger
import re
import torch
import logging

from rxitect.utils.tokenizer import Tokenizer
from rxitect.utils.common import flatten_iterable
from typing import *

logger = logging.getLogger(name=__name__)


class SmilesTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, mol_strings: List[str]) -> None:
        """
        """
        unique_tokens = set(flatten_iterable([self.tokenize(mol_string) for mol_string in mol_strings]))
        self.vocabulary = self.sentinel_tokens + list(unique_tokens)

        assert self.vocabulary[0] == self.pad_token, "Pad token has to be at index '0'"

        self.num_tokens = len(self.vocabulary)
        self.stoi = dict(zip(self.vocabulary, range(self.num_tokens)))
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def tokenize(cls, mol_string: str) -> List[str]:
        """Method that takes the string representation of a molecule
        (in this case, SMILES) and returns a list of tokens that the string
        is made up of.
        Args:
            mol_string (str): The SMILES representation of a molecule
        Returns:
            A list of tokens that make up the passed SMILES.
        """
        tokens = []
        regex = '(\[[^\[\]]{1,6}\])'
        mol_string = re.sub('\[\d+', '[', mol_string)
        mol_string = mol_string.replace('Br', 'R').replace('Cl', 'L')
        for word in re.split(regex, mol_string):
            if word == '' or word is None:
                continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for char in word:
                    tokens.append(char)
        return tokens
    
    @classmethod
    def detokenize(cls, tokenized_mol_string: List[str]) -> str:
        """Takes an array of indices and returns the corresponding SMILES"""
        smiles = "".join(tokenized_mol_string)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def encode(self, mol_string: str) -> torch.Tensor:
        """Takes a list containing tokens from the passed SMILES (eg '[NH]') and encodes to array of indices"""
        tokenized_mol_string = [self.start_token] + self.tokenize(mol_string) + [self.end_token]
        encoded_smiles = torch.zeros(size=(1, 128), dtype=torch.long)
        for index, token in enumerate(tokenized_mol_string):
            encoded_smiles[0, index] = self.stoi.get(token, self.stoi[self.missing_token])
        return encoded_smiles

    def batch_encode(self, mol_strings: List[str]) -> torch.Tensor:
        tokenized_mol_strings = [[self.start_token] + self.tokenize(mol_string) + [self.end_token] for mol_string in mol_strings]
        encoded_smiles = torch.zeros(size=(len(tokenized_mol_strings), 128), dtype=torch.long)
        for i, tokenized_smiles in tokenized_mol_strings:
            for j, token in enumerate(tokenized_smiles):
                encoded_smiles[i, j] = self.stoi.get(token, self.stoi[self.missing_token])
        return encoded_smiles

    def decode(self, mol_tensor: torch.Tensor) -> str:
        """Takes an array of indices and returns the corresponding SMILES"""
        tokens = [self.itos[index] for index in mol_tensor]
        smiles = self.detokenize(tokens)
        smiles = smiles.strip(self.start_token, self.end_token)
        return smiles
