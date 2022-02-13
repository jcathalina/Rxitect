from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
from typing import List, Tuple

import selfies as sf
import torch
from attr import field
from rdkit import Chem

from globals import root_path


@dataclass
class Vocabulary(ABC):
    vocabulary_file_path: os.PathLike
    max_len: int = 100
    control: List[str] = field(init=False)
    words: List[str] = field(init=False)
    size: int = field(init=False)
    tk2ix: dict = field(init=False)
    ix2tk: dict = field(init=False)

    def __post_init__(self):
        self.control = ["EOS", "GO"]
        self.words = self.control + self.from_file(self.vocabulary_file_path)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(self.size)))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}

    @classmethod
    @abstractmethod
    def tokenize(cls, selfie: str) -> List[str]:
        pass

    def encode(self, tokenized_mol_str: List[List[str]]) -> torch.Tensor:
        tokens = torch.zeros(len(tokenized_mol_str), self.max_len, dtype=torch.long)
        for i, selfie in enumerate(tokenized_mol_str):
            for j, char in enumerate(selfie):
                tokens[i, j] = self.tk2ix[char]
        return tokens

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = []
        for i in tensor:
            token = self.ix2tk[i.item()]
            if token == "EOS":
                break
            if token in self.control:
                continue
            tokens.append(token)
        mol_str = "".join(tokens)
        return mol_str

    @classmethod
    def from_file(cls, file_path: os.PathLike) -> List[str]:
        with open(file=file_path, mode="r") as f:
            chars = f.read().split()
            words = list(sorted(set(chars)))
        return words

    @abstractmethod
    def check_smiles(self, sequences) -> bool:
        pass


@dataclass
class SelfiesVocabulary(Vocabulary):
    max_len: int = 109  # longest SELFIES in ChEMBL

    @classmethod
    def tokenize(cls, selfie: str) -> List[str]:
        selfie_tokens = sf.split_selfies(selfie)
        return list(selfie_tokens)

    def check_smiles(self, sequences) -> Tuple[List[str], List[int]]:
        selfies = [self.decode(s) for s in sequences]
        smiles = [sf.decoder(selfie) for selfie in selfies]
        valids = [1 if Chem.MolFromSmiles(smile) else 0 for smile in smiles]
        return smiles, valids


if __name__ == "__main__":
    v = SelfiesVocabulary(
        vocabulary_file_path=root_path / "data/processed/selfies_voc.txt"
    )
    benzene = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
    benzene_tokens = [v.tokenize(benzene)]
    print(benzene_tokens)
    benze_tensor = v.encode(benzene_tokens)
    print(benze_tensor)
    print(len(benze_tensor[0]))
    benzene_decoded = v.decode(benze_tensor[0])
    print(benzene_decoded)
