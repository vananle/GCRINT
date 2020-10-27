import numpy as np
import torch


# Loss function
def partial_mse(X, Xhat, W, Wo, scaler=None, lamda=2.0):
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

    err = Xhat - X
    return torch.sum(Wo * W * (err) ** 2) / (torch.sum(Wo * W) + 1e-8)


def partial_mse_u(X, Xhat, W, Wo, scaler=None, lamda=2.0):
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

    err = Xhat - X
    err[err < 0.0] = err[err < 0.0] * lamda

    return torch.sum(Wo * W * (err) ** 2) / (torch.sum(Wo * W) + 1e-8)


def partial_mae(X, Xhat, W, Wo, scaler=None, lamda=2.0):
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

    err = Xhat - X

    return torch.sum(torch.abs(Wo * W * (err))) / (torch.sum(Wo * W) + 1e-8)


def partial_mae_u(X, Xhat, W, Wo, scaler=None, lamda=2.0):
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

    err = Xhat - X
    err[err < 0.0] = err[err < 0.0] * lamda

    return torch.sum(torch.abs(Wo * W * (err))) / (torch.sum(Wo * W) + 1e-8)


def calculate_metrics(X, M, W, Wo):
    _, _rse = rse(X, M, W, Wo)
    _, _mae = mae(X, M, W, Wo)
    _, _mape = mape(X, M, W, Wo)
    _, _mse = mse(X, M, W, Wo)
    _, _rmse = rmse(X, M, W, Wo)

    return _rse, _mae, _mape, _mse, _rmse


# RSE performance metric
def rse(X, M, W, Wo):
    """
    Calculate relative square error
    :param X: GT DATA
    :param M:
    :param W:
    :param Wo:
    :return:
    """
    sample_rse = torch.sqrt(torch.sum(W * Wo * (X - M) ** 2) / (torch.sum(W * Wo * X ** 2) + 1e-8))
    infer_rse = torch.sqrt(torch.sum((1 - W * Wo) * (X - M) ** 2) / (torch.sum((1 - W * Wo) * X ** 2) + 1e-8))
    return float(sample_rse), float(infer_rse)


# MAPE performance metric
def mape(X, M, W, Wo):
    eps = 1e-8
    X = torch.clamp_min(X, eps)
    sample_mape = torch.sum(W * Wo * torch.abs((X - M) / X)) / (torch.sum(W * Wo) + 1e-8)
    infer_mape = torch.sum((1 - W * Wo) * torch.abs((X - M) / X)) / (torch.sum(1 - W * Wo) + 1e-8)
    return float(sample_mape), float(infer_mape)


# MSE performance metric
def mse(X, M, W, Wo):
    sample_mse = torch.sum(W * Wo * (X - M) ** 2) / (torch.sum(W * Wo) + 1e-8)
    infer_mse = torch.sum((1 - W * Wo) * (X - M) ** 2) / (torch.sum(1.0 - W * Wo) + 1e-8)
    return float(sample_mse), float(infer_mse)


# RMSE performance metric
def rmse(X, M, W, Wo):
    sample_rmse = torch.sqrt(torch.sum(W * Wo * (X - M) ** 2) / (torch.sum(W * Wo) + 1e-8))
    infer_rmse = torch.sqrt(torch.sum((1 - W * Wo) * (X - M) ** 2) / (torch.sum(1.0 - W * Wo) + 1e-8))
    return float(sample_rmse), float(infer_rmse)


# MAE performance metric
def mae(X, M, W, Wo):
    sample_mae = torch.sum(W * Wo * torch.abs(X - M)) / (torch.sum(W * Wo) + 1e-8)
    infer_mae = torch.sum((1 - W * Wo) * torch.abs(X - M)) / (torch.sum(1.0 - W * Wo) + 1e-8)
    return float(sample_mae), float(infer_mae)


def mape_np(X, M, W, Wo):
    eps = 1e-8
    X[X <= 0] = eps
    sample_mape = np.sum(W * Wo * np.abs((X - M) / X)) / (np.sum(W * Wo) + 1e-8)
    infer_mape = np.sum((1 - W * Wo) * np.abs((X - M) / X)) / (np.sum(1 - W * Wo) + 1e-8)
    return float(sample_mape), float(infer_mape)


def rmse_np(X, M, W, Wo):
    sample_rmse = np.sqrt(np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo) + 1e-8))
    infer_rmse = np.sqrt(np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum(1.0 - W * Wo) + 1e-8))
    return float(sample_rmse), float(infer_rmse)


def mae_np(X, M, W, Wo):
    sample_mae = np.sum(W * Wo * np.abs(X - M)) / (np.sum(W * Wo) + 1e-8)
    infer_mae = np.sum((1 - W * Wo) * np.abs(X - M)) / (np.sum(1.0 - W * Wo) + 1e-8)
    return float(sample_mae), float(infer_mae)


# RSE numpy performance metric
def rse_np(X, M, W, Wo):
    sample_rse = np.sqrt(np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo * X ** 2) + 1e-8))
    infer_rse = np.sqrt(np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum((1 - W * Wo) * X ** 2) + 1e-8))

    return float(sample_rse), float(infer_rse)


# MSE numpy performance metric
def mse_np(X, M, W, Wo):
    sample_mse = np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo) + 1e-8)
    infer_mse = np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum(1.0 - W * Wo) + 1e-8)
    return float(sample_mse), float(infer_mse)
