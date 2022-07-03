from typing import Tuple
from pytorch_lightning import LightningModule
import torch
from torch import Tensor, nn
from torch import optim
from pathlib import Path

from rxitect.utils.smiles import SmilesTokenizer


class SmilesLSTM(LightningModule):
    def __init__(self,
                 vocabulary_filepath: Path,
                 embed_size: int = 128,
                 hidden_size: int = 512,
                 lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = SmilesTokenizer()
        self.tokenizer.fit_from_file(vocabulary_filepath)

        self.output_size = self.tokenizer.num_tokens

        self.embed = nn.Embedding(self.tokenizer.num_tokens, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.tokenizer.num_tokens)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.embed(x.unsqueeze(-1))
        out, h_out = self.rnn(out, h)
        out = self.linear(out).squeeze(1)
        return out, h_out

    def training_step(self, batch, batch_idx) -> float:
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx) -> float:
        loss = self.likelihood(batch)
        loss = -loss.mean()
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512, device=self.device)
        if labels is not None:
            h[0, batch_size, 0] = labels
        c = torch.rand(3, batch_size, self.hparams.hidden_size, device=self.device) 
        return (h, c)

    def likelihood(self, target: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = target.size()
        x = torch.tensor([self.tokenizer.stoi[self.tokenizer.start_token]] * batch_size, dtype=torch.long, device=self.device)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len, device=self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step : step + 1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores