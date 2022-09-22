from typing import Tuple
import torch
import torch.nn as nn

from rxitect.tokenizers import get_tokenizer


class LSTMGenerator(nn.Module):
    """
    A molecule generator that uses an LSTM to learn how to build valid molecular representations
    through BPTT.

    Attributes
    ----------
    tokenizer : Tokenizer
        A tokenizer to handle a given molecular representation (e.g., SMILES or SELFIES).
    embedding_size : int
        TODO
    hidden_size : int
        TODO
    embedding_layer : torch.nn.Embedding
        TODO
    lstm : torch.nn.LSTM
        TODO
    output_layer : torch.nn.Linear
        TODO
    device : torch.device
        TODO
    """

    def __init__(
        self,
        vocabulary_filepath: str,
        molecule_repr: str = "smiles",
        max_output_len: int = 100,
        embedding_size: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        vocabulary_filepath : str
            TODO
        molecule_repr : str, optional
            The type of molecular (string) representation to use (default is "smiles")
        max_output_len : int, optional
            TODO
        embedding_size : int, optional
            The size of the embedding layer (default is 128)
        hidden_size : int, optional
            The size of the hidden layer (default is 512)
        num_layers : int
            TODO
        device : str
            TODO
        """
        super().__init__()
        self.tokenizer = get_tokenizer(
            molecule_repr, vocabulary_filepath=vocabulary_filepath, max_len=max_output_len
        )
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.tokenizer.vocabulary_size_, embedding_dim=embedding_size
        )
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, self.tokenizer.vocabulary_size_)

        # Store base device of model as attribute and send model if necessary.
        self.device = torch.device(device)
        self.to(self.device)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding_layer(x.unsqueeze(dim=-1))
        x, h_out = self.lstm(x, h)
        x = self.output_layer(x).squeeze(dim=1)
        return x, h_out

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.rand(3, batch_size, self.hidden_size, device=self.device)
        c = torch.rand(3, batch_size, self.hidden_size, device=self.device)
        return h, c

    def likelihood(self, target: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = target.size()
        x = torch.LongTensor(
            [self.tokenizer.tk2ix_[self.tokenizer.start_token]] * batch_size,
            device=self.device,
        )
        h = self.init_hidden(batch_size)
        scores = torch.zeros(batch_size, seq_len, device=self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step : step + 1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores


class GRUGenerator(nn.Module):
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
            num_embeddings=self.tokenizer.vocabulary_size_, embedding_dim=embedding_size
        )
