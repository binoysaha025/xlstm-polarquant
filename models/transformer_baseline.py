import torch
import torch.nn as nn

class TransformerBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size*4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, states=None):
        x = self.input_proj(x)           # (batch, seq, hidden)
        x = self.transformer(x)          # (batch, seq, hidden)
        out = self.output_head(x[:, -1, :])  # take last token
        return out, None