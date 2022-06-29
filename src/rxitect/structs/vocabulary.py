import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union
import pandas as pd

import selfies as sf
import torch
from attr import field
from rdkit import Chem


@dataclass
class Vocabulary(ABC):
    vocabulary_filepath: Optional[os.PathLike] = None
    max_len: int = 100
    sentinel_tokens: List[str] = field(init=False)
    start: str = "^"
    end: str = "$"
    unk: str = "?"
    pad: str = " "

    tokens: List[str] = field(init=False)
    size: int = field(init=False)
    tk2ix: dict = field(init=False)
    ix2tk: dict = field(init=False)

    def __post_init__(self):
        self.sentinel_tokens = [self.pad, self.start, self.end, self.unk]
        self.tokens = (
            self.sentinel_tokens + self.from_file(self.vocabulary_filepath)
            if self.vocabulary_filepath
            else self.sentinel_tokens
        )
        self.size = len(self.tokens)
        self.tk2ix = dict(zip(self.tokens, range(self.size)))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}

    @classmethod
    @abstractmethod
    def tokenize(cls, selfie: str) -> List[str]:
        pass
    
    def encode(self, smiles: str) -> List[int]:
        return [self.tk2ix[self.start]] + [self.tk2ix.get(char, self.tk2ix[self.unk]) for char in smiles] + [self.tk2ix[self.end]]

    def decode(self, vectorized_mol: Union[torch.Tensor, List[str]]) -> List[str]:
        return [self.ix2tk[i] for i in vectorized_mol]


    @classmethod
    def from_file(cls, filepath: os.PathLike) -> List[str]:
        with open(file=filepath, mode="r") as f:
            chars = f.read().split()
            tokens = list(sorted(set(chars)))
        return tokens

    def fit(self, smiles: List[str]) -> None:
        charset = set("".join(list(smiles)))
        #Important that pad gets value 0
        self.tokens = self.sentinel_tokens + list(charset)
        self.size = len(self.tokens)
        self.tk2ix = dict(zip(self.tokens, range(self.size)))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}

    @abstractmethod
    def check_validity(self, sequences) -> bool:
        pass


@dataclass
class SelfiesVocabulary(Vocabulary):
    max_len: int = 109  # longest SELFIES in ChEMBL v26  # TODO: Check what this is for v30 now...
    mol_str_type = "selfies"

    @classmethod
    def tokenize(cls, selfie: str) -> List[str]:
        selfie_tokens = sf.split_selfies(selfie)
        return list(selfie_tokens)

    def check_validity(self, sequences: torch.Tensor) -> Tuple[List[str], List[int]]:
        selfies = [self.decode(s) for s in sequences]
        smiles = [sf.decoder(selfie) for selfie in selfies]
        valids = [1 if Chem.MolFromSmiles(smile) else 0 for smile in smiles]
        return smiles, valids


@dataclass
class SmilesVocabulary(Vocabulary):
    mol_str_type = "smiles"

    @classmethod
    def tokenize(cls, smiles: str) -> List[str]:
        """
        Takes a SMILES and return a list of characters/tokens
        Args:
            smiles (str): a decoded smiles sequence.
        Returns:
            tokens (List[str]): a list of tokens decoded from the SMILES sequence.
        """
        regex = "(\[[^\[\]]{1,6}\])"
        smile = smiles.replace("Cl", "L").replace("Br", "R")
        tokens = []
        for word in re.split(regex, smile):
            if word == "" or word is None:
                continue
            if word.startswith("["):
                tokens.append(word)
            else:
                for char in word:
                    tokens.append(char)
        return tokens

    def check_validity(
        self, seqs: Iterable[torch.LongTensor]
    ) -> Tuple[List[str], List[int]]:
        """
        Args:
            seqs (Iterable[torch.LongTensor]): a batch of token indices.
        Returns:
            smiles (List[str]): a list of decoded SMILES
            valids (List[int]): if the decoded SMILES is valid or not
        """
        smiles = [self.decode(s) for s in seqs]
        valids = [1 if Chem.MolFromSmiles(smile) else 0 for smile in smiles]
        return smiles, valids
