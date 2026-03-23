# models/decoder.py

import torch.nn as nn
from .attention import scaled_dot_product_attention

class DecoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, y, encoder_output, mask):
        # 🔥 self-attention com máscara correta
        attn1, _ = scaled_dot_product_attention(y, y, y, mask)

        # 🔥 cross-attention (SEM máscara causal)
        attn2, _ = scaled_dot_product_attention(attn1, encoder_output, encoder_output, None)

        out = self.ffn(attn2)
        return out


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model) for _ in range(num_layers)
        ])

    def forward(self, y, encoder_output, mask):
        for layer in self.layers:
            y = layer(y, encoder_output, mask)
        return y