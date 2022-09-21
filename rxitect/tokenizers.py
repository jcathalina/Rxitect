from typing import List, Optional
import torch

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    vocabulary_size: int

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
        SENTINEL_TOKENS = ["GO", "EOS", "nop"]

        self.start_token = SENTINEL_TOKENS[0]
        self.stop_token = SENTINEL_TOKENS[1]
        self.nop_token = SENTINEL_TOKENS[2]
        self.vocabulary = SENTINEL_TOKENS + self._get_vocabulary_from_file(
            vocabulary_filepath
        )
        self.max_len = max_len
        self.vocabulary_size_ = len(self.vocabulary)
        self.tk2ix_ = dict(zip(self.vocabulary, range(self.vocabulary_size_)))
        self.ix2tk_ = {ix: tk for tk, ix in self.tk2ix_.items()}

    def encode(self, molecules: List[str]) -> torch.Tensor:
        print("Encoding some SMILES!")
        encoded_smiles = torch.zeros(len(molecules), self.max_len, dtype=torch.long)
        for i, smi in enumerate(molecules):
            for j, token in enumerate(smi):
                encoded_smiles[i, j] = self.tk2ix_[token]
        return encoded_smiles

    def decode(self, encoded_molecules: torch.Tensor) -> List[str]:
        print("Decoding some tensors to SMILES!")


class SelfiesTokenizer(Tokenizer):
    def __init__(self, vocabulary_filepath: str, max_len: int) -> None:
        SENTINEL_TOKENS = ["[GO]", "[EOS]", "[nop]"]

        self.start_token = SENTINEL_TOKENS[0]
        self.stop_token = SENTINEL_TOKENS[1]
        self.nop_token = SENTINEL_TOKENS[2]
        self.vocabulary = SENTINEL_TOKENS + self._get_vocabulary_from_file(
            vocabulary_filepath
        )
        self.vocabulary_size_ = len(self.vocabulary)
        self.max_len = max_len
        self.tk2ix_ = dict(zip(self.vocabulary, range(self.vocabulary_size_)))
        self.ix2tk_ = {ix: tk for tk, ix in self.tk2ix_.items()}

    def encode(molecules: List[str]) -> torch.Tensor:
        print("Encoding some SELFIES!")

    def decode(encoded_molecules: torch.Tensor) -> List[str]:
        print("Decoding some tensors to SELFIES!")


def get_tokenizer(molecule_repr: str, vocabulary_filepath: str, max_len: int) -> Tokenizer:
    if molecule_repr == "smiles":
        return SmilesTokenizer(vocabulary_filepath=vocabulary_filepath, max_len=max_len)
    elif molecule_repr == "selfies":
        return SelfiesTokenizer(vocabulary_filepath=vocabulary_filepath, max_len=max_len)
    else:
        raise ValueError(molecule_repr)
