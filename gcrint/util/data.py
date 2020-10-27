import os

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from .interp import interp


class MissingDataset(Dataset):

    def __init__(self, X, W, Wo, args):
        # load observed data
        self.args = args

        X_obs = X * W * Wo

        # load interpole data
        X_interpole = interp(X_obs, W)
        self.X_linear_imp = np.copy(X_interpole)
        # learn scaler
        self.scaler = MinMaxScaler().fit(X_interpole)

        # transform if needed and convert to torch
        X_interpole = self.transform(X_interpole)

        self.X_interpole = self.np2torch(X_interpole)
        self.X = self.np2torch(X)
        self.Wo = self.np2torch(Wo)
        self.W = self.np2torch(W)
        self.X_linear_imp = self.np2torch(self.X_linear_imp)
        # save parameters

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.X_interpole[t:t + self.args.seq_len - 2]
        w = self.W[t:t + self.args.seq_len - 2]

        x_inv = self.X_interpole[t + 2:t + self.args.seq_len]
        w_inv = self.W[t + 2:t + self.args.seq_len]
        x_inv = torch.flip(x_inv, dims=[0])
        w_inv = torch.flip(w_inv, dims=[0])

        y = self.X_interpole[t + 1:t + self.args.seq_len - 1]
        wy = self.Wo[t + 1:t + self.args.seq_len - 1]

        sample = {'x': x,
                  'w': w,
                  'xi': x_inv,
                  'wi': w_inv,
                  'y': y,
                  'wy': wy}
        return sample

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        T, D = self.W.shape
        indices = np.arange(T - self.args.seq_len)
        return indices


def load_matlab_matrix(path, variable_name):
    X = loadmat(path)[variable_name]
    return X


def load_raw(args):
    # load ground truth
    path = os.path.join(args.datapath, 'data/{}.mat'.format(args.dataset))
    X = load_matlab_matrix(path, 'X')
    try:
        Wo = load_matlab_matrix(path, 'Wo')
    except:
        Wo = np.ones_like(X)

    # load missing mask
    path = os.path.join(args.datapath, 'mask/{}/{}/{:0.1f}_{}.mat'.format(args.dataset, args.type, args.sr, args.seed))
    W = load_matlab_matrix(path, 'W')

    if X.shape[0] >= 10000:
        X = X[:10000]
        W = W[:10000]
        Wo = Wo[:10000]

    return X, W, Wo


def create_validation_mask(W, args):
    validation_mask = np.random.choice([1, 0], size=W.shape,
                                       p=[1 - args.p_validation, args.p_validation])
    Wval = W * validation_mask
    return Wval


def data_splitting(X, W, Wo):
    train_size = int(X.shape[0] * 0.7)
    val_size = int(X.shape[0] * 0.1)

    X_train = X[:train_size]
    W_train = W[:train_size]
    Wo_train = Wo[:train_size]

    X_val = X[train_size:val_size + train_size]
    W_val = W[train_size:val_size + train_size]
    Wo_val = Wo[train_size:val_size + train_size]

    X_test = X[val_size + train_size:]
    W_test = W[val_size + train_size:]
    Wo_test = Wo[val_size + train_size:]

    return (X_train, W_train, Wo_train), (X_val, W_val, Wo_val), (X_test, W_test, Wo_test)


def get_dataloader(args):
    # loading data
    X, W, Wo = load_raw(args)

    # data (X, W, Wo)
    train_data, valid_data, test_data = data_splitting(X, W, Wo)  # split data as 70-10-20

    Wval_train = create_validation_mask(train_data[1], args)
    Wval_val = create_validation_mask(valid_data[1], args)
    Wval_test = create_validation_mask(test_data[1], args)

    # Training set for imputation
    train_dataset = MissingDataset(train_data[0], train_data[1], train_data[2], args)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.imp_batch_size,
                                  shuffle=True)

    # train validation set for imputation
    train_val_set = MissingDataset(train_data[0], Wval_train, train_data[2], args)
    train_val_loader = DataLoader(train_val_set,
                                  batch_size=args.imp_batch_size,
                                  shuffle=False)

    # val set for imputation
    val_dataset = MissingDataset(valid_data[0], valid_data[1], valid_data[2], args)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.imp_batch_size,
                                shuffle=True)

    # val validation set for imputation
    val_val_set = MissingDataset(valid_data[0], Wval_val, valid_data[2], args)
    val_val_loader = DataLoader(val_val_set,
                                batch_size=args.imp_batch_size,
                                shuffle=False)

    # test set for imputation
    test_dataset = MissingDataset(test_data[0], test_data[1], test_data[2], args)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.imp_batch_size,
                                 shuffle=True)

    # test validation set for imputation
    test_val_set = MissingDataset(test_data[0], Wval_test, test_data[2], args)
    test_val_loader = DataLoader(test_val_set,
                                 batch_size=args.imp_batch_size,
                                 shuffle=False)

    # return (dataloader, val_dataloader) for each set
    return (train_dataloader, train_val_loader), (val_dataloader, val_val_loader), (test_dataloader, test_val_loader)
