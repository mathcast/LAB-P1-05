import torch.nn as nn
from models.attention import SelfAttention
from models.add_norm import AddNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attention = SelfAttention(d_model)

        self.add_norm1 = AddNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.add_norm2 = AddNorm(d_model)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.add_norm1(x, attn_out)

        ff_out = self.ff(x)
        x = self.add_norm2(x, ff_out)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x