import re
from abc import ABC, abstractmethod
from typing import List

import selfies as sf
import torch


class Tokenizer(ABC):
    vocabulary_size_: int

    @abstractmethod
    def encode(self, molecules: List[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, encoded_molecules: torch.Tensor) -> List[str]:
        pass

    def _get_vocabulary_from_file(self, vocabulary_filepath: str) -> List[str]:
        with open(vocabulary_filepath, "r") as f:
            vocabulary = f.read().splitlines()

        return sorted(vocabulary)


class SmilesTokenizer(Tokenizer):
    def __init__(self, vocabulary_filepath: str, max_len: int) -> None:
        self.start_token = "GO"
        self.stop_token = "EOS"
        self.pad_token = " "
        SENTINEL_TOKENS = [self.start_token, self.stop_token, self.pad_token]
        self.vocabulary = SENTINEL_TOKENS + self._get_vocabulary_from_file(
            vocabulary_filepath
        )
        self.max_len = max_len
        self.vocabulary_size_ = len(self.vocabulary)
        self.tk2ix_ = dict(zip(self.vocabulary, range(self.vocabulary_size_)))
        self.ix2tk_ = {ix: tk for tk, ix in self.tk2ix_.items()}

    def encode(self, molecule: str) -> torch.Tensor:
        print("Encoding single SMILES!")
        encoded_smiles = torch.zeros(self.max_len, dtype=torch.long)
        tokenized_smiles = self._tokenize(molecule)
        for i, token in enumerate(tokenized_smiles):
            encoded_smiles[i] = self.tk2ix_[token]
        return encoded_smiles

    def batch_encode(self, molecules: List[str]) -> torch.Tensor:
        print("Encoding some SMILES!")
        encoded_smiles = torch.zeros(len(molecules), self.max_len, dtype=torch.long)
        for i, smi in enumerate(molecules):
            tokenized_smi = self._tokenize(smi)
            for j, token in enumerate(tokenized_smi):
                encoded_smiles[i, j] = self.tk2ix_[token]
        return encoded_smiles

    def decode(self, encoded_molecule: torch.Tensor) -> List[str]:
        print("Decoding single tensor to SMILES!")
        encoded_molecule = encoded_molecule.cpu().detach().numpy()
        chars = []
        for i in encoded_molecule:
            if i == self.tk2ix_[self.stop_token]:
                break
            chars.append(self.ix2tk_[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def batch_decode(self, encoded_molecules: torch.Tensor) -> List[str]:
        print("Decoding some tensors to SMILES!")
        decoded_smiles = []
        encoded_molecules = encoded_molecules.cpu().detach().numpy()
        for enc_smiles in encoded_molecules:
            chars = []
            for i in enc_smiles:
                if i == self.tk2ix_[self.stop_token]:
                    break
                chars.append(self.ix2tk_[i])
            smiles = "".join(chars)
            smiles = smiles.replace("L", "Cl").replace("R", "Br")
            decoded_smiles.append(smiles)
        return decoded_smiles

    def _tokenize(self, smiles: str) -> List[str]:
        """
        Takes a SMILES string and returns a list containing the tokens its composed of.
        SOURCE: https://github.com/MarcusOlivecrona/REINVENT/

        Parameters
        ----------
        smiles: A SMILES string representing a molecule
        """
        regex = "(\[[^\[\]]{1,6}\])"
        smiles = self._replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith("["):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append(self.stop_token)
        return tokenized

    def _replace_halogen(self, smiles: str) -> str:
        """Regex to replace Br and Cl with single letters"""
        br = re.compile("Br")
        cl = re.compile("Cl")
        smiles = br.sub("R", smiles)
        smiles = cl.sub("L", smiles)

        return smiles


class SelfiesTokenizer(Tokenizer):
    def __init__(self, vocabulary_filepath: str, max_len: int) -> None:
        self.start_token = "[GO]"
        self.stop_token = "[EOS]"
        self.pad_token = "[nop]"
        SENTINEL_TOKENS = [self.pad_token, self.start_token, self.stop_token]
        self.vocabulary = SENTINEL_TOKENS + self._get_vocabulary_from_file(
            vocabulary_filepath
        )
        self.vocabulary_size_ = len(self.vocabulary)
        self.max_len = max_len
        self.tk2ix_ = dict(zip(self.vocabulary, range(self.vocabulary_size_)))
        self.ix2tk_ = {ix: tk for tk, ix in self.tk2ix_.items()}

    def encode(self, molecule: List[str]) -> torch.Tensor:
        print("Encoding some SELFIES!")
        encoded_smiles = torch.zeros(self.max_len, dtype=torch.long)
        tokenized_smiles = self._tokenize(molecule)
        for i, token in enumerate(tokenized_smiles):
            encoded_smiles[i] = self.tk2ix_[token]
        return encoded_smiles

    def decode(self, encoded_molecule: torch.Tensor) -> List[str]:
        print("Decoding some tensors to SELFIES!")
        encoded_molecule = encoded_molecule.cpu().detach().numpy()
        chars = []
        for i in encoded_molecule:
            if i == self.tk2ix_[self.stop_token]:
                break
            chars.append(self.ix2tk_[i])
        selfies = "".join(chars)
        return selfies

    def _tokenize(self, selfies: str) -> List[str]:
        """
        Takes a SELFIES string and returns a list containing the tokens its composed of.

        Parameters
        ----------
        selfies: A SELFIES string representing a molecule
        """
        tokenized_selfies = list(sf.split_selfies(selfies))
        tokenized_selfies.append(self.stop_token)
        return tokenized_selfies


def get_tokenizer(
    molecule_repr: str, vocabulary_filepath: str, max_len: int
) -> Tokenizer:
    if molecule_repr == "smiles":
        return SmilesTokenizer(vocabulary_filepath=vocabulary_filepath, max_len=max_len)
    elif molecule_repr == "selfies":
        return SelfiesTokenizer(
            vocabulary_filepath=vocabulary_filepath, max_len=max_len
        )
    else:
        raise ValueError(molecule_repr)
