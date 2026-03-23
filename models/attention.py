import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        print("\nATTENTION DEBUG")
        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        print("Scores:", scores)

        attention = F.softmax(scores, dim=-1)

        print("Attention:", attention)

        output = torch.matmul(attention, V)

        return output