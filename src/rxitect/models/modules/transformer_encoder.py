import torch
from torch import nn

from rxitect.models.modules.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int = 110,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=n_tokens)

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
