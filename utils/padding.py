import torch

def pad_sequence(seq, max_len, pad_token):
    return seq + [pad_token] * (max_len - len(seq))


def create_look_ahead_mask(size):
    return torch.tril(torch.ones(size, size))