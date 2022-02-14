from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from rxitect.structs.vocabulary import Vocabulary
from rxitect import tensor_utils


class Generator(pl.LightningModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        embed_size: int = 128,
        hidden_size: int = 512,
        lr: float = 1e-3,
        is_lstm: bool = True,
    ):
        """
        Class defining the molecule generating LSTM model.

        Args:
            vocabulary (Vocabulary): An object containing all tokens available to generate SMILES with.
            embed_size (int): Size of the embedding space.
            hidden_size (int): Number of nodes in the hidden layers.
            lr (float): Learning rate for training, 1e-3 by default.
        """
        super().__init__()
        self.voc = vocabulary
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = vocabulary.size
        self.lr = lr

        self.embed = torch.nn.Embedding(vocabulary.size, embed_size, device=self.device)
        self.rnn_layer = torch.nn.LSTM if is_lstm else torch.nn.GRU
        self.rnn = self.rnn_layer(
            embed_size, hidden_size, num_layers=3, batch_first=True, device=self.device
        )
        self.linear = torch.nn.Linear(hidden_size, vocabulary.size, device=self.device)
        self.automatic_optimization = False

    def forward(self, x, h):
        output = self.embed(x.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes a hidden state for the LSTM
        Args:
            batch_size (int): batch size to use for the hidden state.

        Returns:
            A hidden state tuple
        """
        h = torch.rand(3, batch_size, 512, device=self.device)
        c = torch.rand(3, batch_size, self.hidden_size, device=self.device)
        return h, c

    def likelihood(self, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the likelihood of the next token in the sequence for a batch of molecules
        Args:
            target (torch.Tensor): A tensor containing the size of the batch to use and the associated sequence length

        Returns:
            A tensor containing likelihood scores for all molecules in the batch
        """
        batch_size, seq_len = target.size()
        x = torch.tensor(
            [self.voc.tk2ix["GO"]] * batch_size, dtype=torch.long, device=self.device
        )
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len, device=self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step : step + 1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def policy_gradient_loss(self, loader: DataLoader) -> None:
        """ """
        for sequence, reward in loader:
            self.zero_grad()
            score = self.likelihood(sequence)
            loss = score * reward
            loss = -loss.mean()
            self.manual_backward(loss)
            self.opt.step()

    def sample(self, batch_size):
        """ """
        x = torch.tensor(
            [self.voc.tk2ix["GO"]] * batch_size, dtype=torch.long, device=self.device
        )
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len, device=self.device).long()
        is_end = torch.zeros(batch_size, device=self.device).bool()

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[is_end] = self.voc.tk2ix["EOS"]
            sequences[:, step] = x

            end_token = x == self.voc.tk2ix["EOS"]
            is_end = torch.ge(is_end + end_token, 1)
            if (is_end == 1).all():
                break
        return sequences

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.manual_backward(loss)
        opt.step()
        self.log("train_loss", loss)
        print(f"Train Loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.log("val_loss", loss)
        print(f"Val Loss: {loss}")
        return loss

    def on_validation_epoch_end(self) -> None:
        sequences = self.sample(1024)  # batch size (512 by default) times 2
        indices = tensor_utils.unique(sequences)
        sequences = sequences[indices]
        smiles, valids = self.voc.check_smiles(sequences)
        frac_valid = sum(valids) / len(sequences)
        self.log("frac_valid_smiles", frac_valid)
        print(f"Fraction of valid smiles: {frac_valid}")


def evolve(
    net: Generator,
    batch_size: int,
    epsilon: float = 0.01,
    crover: Optional["Generator"] = None,
    mutate: Optional["Generator"] = None,
):
    """ """
    # Start tokens
    x = torch.tensor(
        [net.voc.tk2ix["GO"]] * batch_size, dtype=torch.long, device=net.device
    )
    # Hidden states initialization for exploitation network
    h = net.init_h(batch_size)
    # Hidden states initialization for exploration network
    h2 = net.init_h(batch_size)
    # Initialization of output matrix
    sequences = torch.zeros(batch_size, net.voc.max_len, device=net.device).long()
    # labels to judge and record which sample is ended
    is_end = torch.zeros(batch_size, device=net.device).bool()

    for step in range(net.voc.max_len):
        is_change = torch.rand(1) < 0.5
        if crover is not None and is_change:
            logit, h = crover(x, h)
        else:
            logit, h = net(x, h)
        proba = logit.softmax(dim=-1)
        if mutate is not None:
            logit2, h2 = mutate(x, h2)
            ratio = torch.rand(batch_size, 1, device=net.device) * epsilon
            proba = (
                logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
            )
        # sampling based on output probability distribution
        x = torch.multinomial(proba, 1).view(-1)

        x[is_end] = net.voc.tk2ix["EOS"]
        sequences[:, step] = x

        # Judging whether samples are end or not.
        end_token = x == net.voc.tk2ix["EOS"]
        is_end = torch.ge(is_end + end_token, 1)
        #  If all of the samples generation being end, stop the sampling process
        if (is_end == 1).all():
            break
    return sequences
