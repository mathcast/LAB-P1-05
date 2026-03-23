import torch.nn as nn
from models.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = Encoder(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_input=None, mask=None):
        x = self.embedding(src)

        enc_out = self.encoder(x)

        out = self.fc_out(enc_out)

        return out