from typing import List, Optional
import torch

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    vocabulary_size: int

    @abstractmethod
    def encode(molecules: List[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(encoded_molecules: torch.Tensor) -> List[str]:
        pass

    def get_vocabulary_from_file(vocabulary_filepath: str) -> List[str]:
        with open(vocabulary_filepath, "r") as f:
            vocabulary = f.read().splitlines()
        
        return vocabulary


class SmilesTokenizer(Tokenizer):
    def __init__(self, vocabulary_filepath: Optional[str] = None, vocabulary: Optional[List[str]] = None) -> None:
        if vocabulary_filepath:
            self.vocabulary = self.get_vocabulary_from_file(vocabulary_filepath)
        elif vocabulary:
            self.vocabulary = vocabulary
        else:
            raise ValueError("Either a file or a list containing the vocabulary is required.")

        self.vocabulary_size = len(self.vocabulary)


    def encode(molecules: List[str]) -> torch.Tensor:
        print("Encoding some SMILES!")

    def decode(encoded_molecules: torch.Tensor) -> List[str]:
        print("Decoding some tensors to SMILES!")


class SelfiesTokenizer(Tokenizer):
    def __init__(self, vocabulary_filepath: Optional[str] = None, vocabulary: Optional[List[str]] = None) -> None:
        if vocabulary_filepath:
            self.vocabulary = self.get_vocabulary_from_file(vocabulary_filepath)
        elif vocabulary:
            self.vocabulary = vocabulary
        else:
            raise ValueError("Either a file or a list containing the vocabulary is required.")

        self.vocabulary_size = len(self.vocabulary)

    def encode(molecules: List[str]) -> torch.Tensor:
        print("Encoding some SELFIES!")

    def decode(encoded_molecules: torch.Tensor) -> List[str]:
        print("Decoding some tensors to SELFIES!")


def get_tokenizer(molecule_repr: str, vocabulary_filepath: str) -> Tokenizer:
    if molecule_repr == "smiles":
        return SmilesTokenizer(vocabulary_filepath=vocabulary_filepath)
    elif molecule_repr == "selfies":
        return SelfiesTokenizer(vocabulary_filepath=vocabulary_filepath)
    else:
        raise ValueError(molecule_repr)


if __name__ == "__main__":
    t = get_tokenizer("smiles")