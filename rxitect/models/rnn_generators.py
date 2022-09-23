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
            molecule_repr,
            vocabulary_filepath=vocabulary_filepath,
            max_len=max_output_len,
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
        x = torch.tensor(
            [self.tokenizer.tk2ix_[self.tokenizer.start_token]] * batch_size,
            device=self.device,
            dtype=torch.long,
        )
        h = self.init_hidden(batch_size)
        scores = torch.zeros(batch_size, seq_len, device=self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            dummy_idx = target[:, step : step + 1]
            score = logits.gather(1, target[:, step : step + 1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def sample(self, batch_size: int):
        x = torch.tensor(
            [self.tokenizer.tk2ix_[self.tokenizer.start_token]] * batch_size,
            dtype=torch.long,
            device=self.device,
        )
        h = self.init_hidden(batch_size)
        sequences = torch.zeros(
            batch_size, self.tokenizer.max_len, dtype=torch.long, device=self.device
        )
        is_end = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.tokenizer.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[is_end] = self.tokenizer.tk2ix_[self.tokenizer.stop_token]
            sequences[:, step] = x

            end_token = x == self.tokenizer.tk2ix_[self.tokenizer.stop_token]
            is_end = torch.ge(is_end + end_token, 1)
            if (is_end == 1).all():
                break
        return sequences


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


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from pyprojroot import here
    from torch import optim
    from torch.utils.data import DataLoader

    from rxitect.data import SelfiesDataset, SmilesDataset
    from rxitect.models import LSTMGenerator
    from rxitect.tokenizers import (SelfiesTokenizer, SmilesTokenizer,
                                    get_tokenizer)

    def smiles_dataloader(smiles_tokenizer) -> DataLoader:
        test_dataset_filepath = here() / "tests/data/test.smi"
        # test_dataset_filepath = here() / "data/processed/chembl_v30_clean.smi"
        dataset = SmilesDataset(
            dataset_filepath=test_dataset_filepath, tokenizer=smiles_tokenizer
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=128,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            collate_fn=SmilesDataset.collate_fn,
        )
        return dataloader

    class Utils:
        def __init__(self):
            self.dev = "cuda"

    utils = Utils()

    lr = 1e-3
    epochs = 1_000

    net = LSTMGenerator(
        vocabulary_filepath=here() / "tests/data/test_smiles_voc.txt",
        device="cuda",
        max_output_len=200,
    )
    dataloader = smiles_dataloader(net.tokenizer)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            print(i)
            optimizer.zero_grad()
            loss_train = net.likelihood(batch.to(utils.dev))
            loss_train = -loss_train.mean()
            loss_train.backward()
            optimizer.step()
            if i % 1 == 0:
                # seqs = net.sample(len(batch * 2))
                info = "Epoch: %d step: %d loss_train: %.3f" % (
                    epoch,
                    i,
                    loss_train.item(),
                )
                print(info)
