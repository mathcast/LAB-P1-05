import torch.nn as nn

def get_loss(pad_token):
    return nn.CrossEntropyLoss(ignore_index=pad_token)