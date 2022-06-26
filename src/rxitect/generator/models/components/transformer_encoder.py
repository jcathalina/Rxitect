import torch

from rxitect.generator.models.components.positional_encoding import PositionalEncoding


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                n_tokens,
                d_model=256,
                nhead=8,
                num_encoder_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(n_tokens, d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.fc_out = torch.nn.Linear(d_model, n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._generate_square_subsequent_mask(x.shape[0])  # .to(self.device)
        embedded = self.embedding(x)
        positional_encoded = self.positional_encoder(embedded)
        encoded = self.encoder(positional_encoded, mask=mask)
        out_2= self.fc_out(encoded)
        return out_2

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
