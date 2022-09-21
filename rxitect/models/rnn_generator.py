import torch.nn as nn
from torch.nn import RNNBase
from rxitect.tokenizers import get_tokenizer


class LSTMGenerator(nn.Module):
    """
    A molecule generator that uses an LSTM to learn how to build valid molecular representations
    through BPTT.

    Attributes
    ----------
    tokenizer : Tokenizer
        A tokenizer to handle a given molecular representation (e.g., SMILES or SELFIES).

    """

    def __init__(
        self,
        molecule_repr: str = "smiles",
        embedding_size: int = 128,
        hidden_size: int = 512,
    ) -> None:
        """
        Parameters
        ----------
        molecule_repr : str
            The type of molecular (string) representation to use (e.g. "smiles")
        embedding_size : int, optional
            The size of the embedding layer (default is 128)
        hidden_size: int, optional
            The sie of the hidden layer (default is 512)
        """
        super().__init__()
        self.tokenizer = get_tokenizer(molecule_repr)
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.tokenizer.vocabulary_size, embedding_dim=embedding_size
        )
