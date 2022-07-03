from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining the interface that can subsequently be used in
    dataset/datamodules/models."""

    def __init__(
        self,
        pad_token: str = " ",
        start_token: str = "[SOS]",
        end_token: str = "[EOS]",
        missing_token: str = "[UNK]",
        vocabulary: Optional[List[str]] = None,
    ) -> None:
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.missing_token = missing_token
        self.sentinel_tokens = [
            pad_token,
            start_token,
            end_token,
            missing_token,
        ]  # Pad should always be first.
        self.vocabulary = self.sentinel_tokens + vocabulary if vocabulary else self.sentinel_tokens
        self.num_tokens = len(self.vocabulary)
        self.stoi = {token: index for index, token in enumerate(self.vocabulary)}
        self.itos = {index: token for index, token in enumerate(self.vocabulary)}

    @abstractmethod
    def fit(self, mol_strings: List[str]) -> None:
        pass

    @classmethod
    @abstractmethod
    def tokenize(cls, mol_string: str) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def detokenize(cls, tokenized_mol_string: List[str]) -> str:
        pass

    @abstractmethod
    def encode(self, mol_string: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, mol_tensor: torch.Tensor) -> str:
        pass
