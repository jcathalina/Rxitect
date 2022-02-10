import copy
import os
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem


class Vocabulary:
    """
    A class for handling encoding/decoding from SMILES to an array of indices
    """

    def __init__(
        self,
        max_len: int,
        vocabulary_path: Optional[os.PathLike] = None,
    ):
        """
        Args:
            max_len: The threshold length of molecules that we will consider
            vocabulary_path (Optional[os.PathLike]): The path to the file containing the vocabulary that should be used.
            If included, will create the Vocabulary instance from that file. Otherwise, a new vocabulary is created
            from the dataset that is processed, e.g. ChEMBL.
        """
        self.control = ["EOS", "GO"]
        self.words = copy.deepcopy(self.control)
        if vocabulary_path:
            self.words += self.from_file(vocabulary_path)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    @classmethod
    def tokenize(cls, smile: str) -> List[str]:
        """
        Takes a SMILES and return a list of characters/tokens
        Args:
            smile (str): a decoded smiles sequence.
        Returns:
            tokens (List[str]): a list of tokens decoded from the SMILES sequence.
        """
        regex = "(\[[^\[\]]{1,6}\])"
        smile = smile.replace("Cl", "L").replace("Br", "R")
        tokens = []
        for word in re.split(regex, smile):
            if word == "" or word is None:
                continue
            if word.startswith("["):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        return tokens

    def encode(self, smiles: List[str]) -> torch.LongTensor:
        """
        Takes a list of tokens (eg '[NH]') and encodes them to an array of indices
        Args:
            smiles (List[str]): a list of SMILES sequences represented as a series of tokens
        Returns:v
            tokens (torch.LongTensor): a long tensor containing all of the indices of given tokens.
        """
        tokens = torch.zeros(len(smiles), self.max_len).long()
        for i, smile in enumerate(smiles):
            for j, char in enumerate(smile):
                tokens[i, j] = self.tk2ix[char]
        return tokens

    def decode(self, tensor: torch.LongTensor) -> str:
        """
        Takes an array of indices and returns the corresponding SMILES
        Args:
            tensor(torch.LongTensor): a long tensor containing all of the indices of given tokens.
        Returns:
            smiles (str): a decoded smiles sequence.
        """
        tokens = []
        for i in tensor:
            token = self.ix2tk[i.item()]
            if token == "EOS":
                break
            if token in self.control:
                continue
            tokens.append(token)
        smiles = "".join(tokens)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def calc_voc_fp(self, smiles: str, prefix=None) -> np.ndarray:
        """
        Args:
            smiles (str): SMILES string that represents a molecule
            prefix (?): TODO: Ask Xuhan how this works
        Returns:
            fps (np.ndarray): A Fingerprint in the form of a 2D numpy array that represents the molecule
        """
        fps = np.zeros((len(smiles), self.max_len), dtype=np.long)
        for i, smile in enumerate(smiles):
            token = self.tokenize(smile)
            if prefix is not None:
                token = [prefix] + token
            if len(token) > self.max_len:
                continue
            fps[i, :] = self.encode(token)
        return fps

    def check_smiles(
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

    @classmethod
    def from_file(cls, filepath: os.PathLike) -> List[str]:
        """Takes a file containing new-line separated characters to initialize the vocabulary"""
        with open(filepath, "r") as f:
            chars = f.read().split()
            words = list(sorted(set(chars)))
        return words
