import numpy as np
import torch

EPS = 1e-8


# Loss function
def mae_u(preds, labels, lamda=2.0):
    err = preds - labels
    err[err < 0.0] = err[err < 0.0] * lamda

    return torch.mean(torch.abs(err))


def mse_u(preds, labels, lamda=2.0):
    err = preds - labels
    err[err < 0.0] = err[err < 0.0] * lamda

    return torch.mean((err) ** 2)


def rse(preds, labels):
    return torch.sum((preds - labels) ** 2) / torch.sum((labels + EPS) ** 2)


def mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))


def mse(preds, labels):
    return torch.mean((preds - labels) ** 2)


def mape(preds, labels):
    return torch.mean(torch.abs((preds - labels) / (labels + EPS)))


def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def calc_metrics(preds, labels):
    return rse(preds, labels), mae(preds, labels), mse(preds, labels), mape(preds, labels), rmse(preds, labels)


def mape_np(X, M, W, Wo):
    X[X <= 0] = EPS
    sample_mape = np.sum(W * Wo * np.abs((X - M) / X)) / (np.sum(W * Wo) + EPS)
    infer_mape = np.sum((1 - W * Wo) * np.abs((X - M) / X)) / (np.sum(1 - W * Wo) + EPS)
    return float(sample_mape), float(infer_mape)


def rmse_np(X, M, W, Wo):
    sample_rmse = np.sqrt(np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo) + EPS))
    infer_rmse = np.sqrt(np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum(1.0 - W * Wo) + EPS))
    return float(sample_rmse), float(infer_rmse)


def mae_np(X, M, W, Wo):
    sample_mae = np.sum(W * Wo * np.abs(X - M)) / (np.sum(W * Wo) + EPS)
    infer_mae = np.sum((1 - W * Wo) * np.abs(X - M)) / (np.sum(1.0 - W * Wo) + EPS)
    return float(sample_mae), float(infer_mae)


# RSE numpy performance metric
def rse_np(X, M, W, Wo):
    sample_rse = np.sqrt(np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo * X ** 2) + EPS))
    infer_rse = np.sqrt(np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum((1 - W * Wo) * X ** 2) + EPS))

    return float(sample_rse), float(infer_rse)


# MSE numpy performance metric
def mse_np(X, M, W, Wo):
    sample_mse = np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo) + EPS)
    infer_mse = np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum(1.0 - W * Wo) + EPS)
    return float(sample_mse), float(infer_mse)
