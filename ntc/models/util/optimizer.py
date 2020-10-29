import torch

def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)
