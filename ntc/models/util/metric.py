import numpy as np
import torch

# Loss function
def partial_mse(X, Xhat, W, scaler=None):
    """
    Calculate partial mse loss
    :param X: missing data
    :param Xhat: predicted (scaled)
    :param W: Missing mask
    :param Wo: Missing in original data
    :param scaler: scaler
    :return: mse loss
    """
    if scaler:
        Xhat = scaler.inverse_transform(Xhat)

    # return torch.mean(Wo * W * (Xhat - X))
    return torch.sum(W * (Xhat - X) ** 2) / (torch.sum(W) + 1e-8)

def correlation_loss(X, M, C, args):
     loss = []
     batch_size = args.batch_size
     nseries = args.dim
     for i in range(nseries):
         for j in range(nseries):
             loss.append(C[:, i, j] * torch.norm(X[:, :, i] - M[:, :, j]))

     loss = torch.stack(loss)
     return torch.mean(loss)

# RSE performance metric
def rse(X, M, W):
    """
    Calculate relative square error
    :param X:
    :param M:
    :param W:
    :param Wo:
    :return:
    """
    sample_rse = torch.sqrt(torch.sum(W * (X - M) ** 2) / (torch.sum(W * X ** 2) + 1e-8))
    infer_rse = torch.sqrt(torch.sum((1 - W) * (X - M) ** 2) / (torch.sum((1 - W) * X ** 2) + 1e-8))
    return float(sample_rse), float(infer_rse)

# MAE performance metric
def mae(X, M, W):
    sample_mae = torch.sum(W * torch.abs(X - M)) / (torch.sum(W) + 1e-8)
    infer_mae = torch.sum((1 - W) * torch.abs(X - M)) / (torch.sum(1.0 - W) + 1e-8)
    return float(sample_mae), float(infer_mae)
