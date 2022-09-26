import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pytorch_lightning import LightningModule


class TransformerGenerator(LightningModule):
    def __init__(
        self,
        n_tokens: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1_024,
        dropout: float = 0.1,
        activation: str = "relu",
        max_lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.max_lr = max_lr
        self.net = TransformerEncoder(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        # assert self.trainer.datamodule is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, batch: torch.Tensor):
        batch = batch.to(self.device)
        preds = self.forward(batch[:-1])
        loss = F.cross_entropy(preds.transpose(0, 1).transpose(1, 2), batch[1:].transpose(0, 1))
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), self.max_lr)
        epochs = self.trainer.max_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=None,
            epochs=epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=1e3,
            final_div_factor=1e3,
            last_epoch=-1,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1_024,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.fc_out = nn.Linear(d_model, n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = _generate_square_subsequent_mask(x.shape[0]).to(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        embedded = self.embedding(x)
        positional_encoded = self.positional_encoder(embedded)
        encoded = self.encoder(positional_encoded, mask=mask)
        out_2 = self.fc_out(encoded)
        return out_2


def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    """Positional Encoding impl.
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)