import torch.optim as optim

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-4)  