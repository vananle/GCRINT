import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler_torch():

    def __init__(self, min=None, max=None, device='cuda:0'):
        self.min = min
        self.max = max
        self.device = device

    def fit(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)

    def transform(self, data):
        _data = data.clone()
        return (_data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.max - self.min + 1e-8)) + self.min


class StandardScaler_torch():

    def __init__(self):
        self.means = 0
        self.stds = 0

    def fit(self, data):
        self.means = torch.mean(data, dim=0)
        self.stds = torch.std(data, dim=0)

    def transform(self, data):
        _data = data.clone()
        data_size = data.size()

        if len(data_size) > 2:
            _data = _data.reshape(-1, data_size[-1])

        _data = (_data - self.means) / (self.stds + 1e-8)

        if len(data_size) > 2:
            _data = _data.reshape(data.size())

        return _data

    def inverse_transform(self, data):
        data_size = data.size()
        if len(data_size) > 2:
            data = data.reshape(-1, data_size[-1])

        data = (data * (self.stds + 1e-8)) + self.means

        if len(data_size) > 2:
            data = data.reshape(data_size)

        return data


class ImpDataset(Dataset):

    def __init__(self, X, imp_X, W, args, scaler=None):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len

        self.X = self.np2torch(X)
        self.imp_X = self.np2torch(imp_X)
        self.W = self.np2torch(W)

        self.n_timeslots, self.n_series = self.X.shape

        # learn scaler
        if scaler is None:
            self.scaler = StandardScaler_torch()
            self.scaler.fit(self.imp_X)
        else:
            self.scaler = scaler

        # transform if needed and convert to torch
        self.imp_X_scaled = self.scaler.transform(self.imp_X)

        if args.tod:
            self.tod = self.get_tod()

        if args.ma:
            self.ma = self.get_ma()

        if args.mx:
            self.mx = self.get_mx()

        # get valid start indices for sub-series
        self.indices = self.get_indices()

        if torch.isnan(self.X).any():
            raise ValueError('Data has Nan')

    def get_tod(self):
        tod = torch.arange(self.n_timeslots, device=self.args.device)
        tod = (tod % self.args.day_size) * 1.0 / self.args.day_size
        tod = tod.repeat(self.n_series, 1).transpose(1, 0)  # (n_timeslot, nseries)
        return tod

    def get_ma(self):
        ma = torch.zeros_like(self.imp_X_scaled, device=self.args.device)
        for i in range(self.n_timeslots):
            if i <= self.args.seq_len_x:
                ma[i] = self.imp_X_scaled[i]
            else:
                ma[i] = torch.mean(self.imp_X_scaled[i - self.args.seq_len_x:i], dim=0)

        return ma

    def get_mx(self):
        mx = torch.zeros_like(self.imp_X_scaled, device=self.args.device)
        for i in range(self.n_timeslots):
            if i == 0:
                mx[i] = self.imp_X_scaled[i]
            elif 0 < i <= self.args.seq_len_x:
                mx[i] = torch.max(self.imp_X_scaled[0:i], dim=0)[0]
            else:
                mx[i] = torch.max(self.imp_X_scaled[i - self.args.seq_len_x:i], dim=0)[0]
        return mx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.imp_X_scaled[t:t + self.args.seq_len_x]  # step: t-> t + seq_x     (imputed x)
        xgt = self.X[t:t + self.args.seq_len_x]  # step: t-> t + seq_x              (ground-truth x)
        w = self.W[t:t + self.args.seq_len_x]  # step: t-> t + seq_x                (mask)

        # t + seq_x -> t + seq_x + seq_y:                                           (ground-truth)
        y = self.X[t + self.args.seq_len_x: t + self.args.seq_len_x + self.args.seq_len_y]
        y_gt = self.X[t + self.args.seq_len_x: t + self.args.seq_len_x + self.args.seq_len_y]

        # appending mask to data x
        x = x.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]
        w = w.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]
        x = torch.cat([x, w], dim=-1)  # [seq_x, n, 2]

        # appending other features
        if self.args.tod:
            tod = self.tod[t:t + self.args.seq_len_x]
            tod = tod.unsqueeze(dim=-1)  # [seq_x, n, 1]
            x = torch.cat([x, tod], dim=-1)  # [seq_x, n, +1]

        if self.args.ma:
            ma = self.ma[t:t + self.args.seq_len_x]
            ma = ma.unsqueeze(dim=-1)  # [seq_x, n, 1]
            x = torch.cat([x, ma], dim=-1)  # [seq_x, n, +1]

        if self.args.mx:
            mx = self.mx[t:t + self.args.seq_len_x]
            mx = mx.unsqueeze(dim=-1)  # [seq_x, n, 1]
            x = torch.cat([x, mx], dim=-1)  # [seq_x, n, +1]

        sample = {'x': x, 'y': y, 'x_gt': xgt, 'y_gt': y_gt}
        return sample

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def np2torch(self, X):
        X = X.astype(np.float)
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        T, D = self.X.shape
        indices = np.arange(T - self.args.seq_len_x - self.args.seq_len_y)
        return indices


def load_matlab_matrix(path, variable_name):
    X = loadmat(path)[variable_name]
    return X


def load_dataset(args, set):
    path = args.datapath

    data_path = os.path.join(path, '{}_{}_{}/{}.mat'.format(args.dataset, args.sr, args.type, set))
    X = load_matlab_matrix(data_path, 'X')
    imp_X = load_matlab_matrix(data_path, 'X_imp')
    W = load_matlab_matrix(data_path, 'W')
    if len(X.shape) > 2:
        X = np.reshape(X, newshape=(X.shape[0], -1))

    return {'X': X, 'imp_X': imp_X, 'W': W}


def load_raw(args):
    train_set = load_dataset(args, set='train')  # {'X':X, 'imp_X': imp_X, 'W': W}
    val_set = load_dataset(args, set='val')
    test_set = load_dataset(args, set='test')

    return train_set, val_set, test_set


# def train_test_split(X):
#     train_size = int(X.shape[0] * 0.7)
#     val_size = int(X.shape[0] * 0.1)
#
#     X_train = X[:train_size]
#
#     X_val = X[train_size:val_size + train_size]
#
#     X_test = X[val_size + train_size:]
#
#     return X_train, X_val, X_test


def get_dataloader(args):
    # loading data
    train, val, test = load_raw(args)

    # Training set
    train_set = ImpDataset(X=train['X'], imp_X=train['imp_X'], W=train['W'], args=args, scaler=None)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True)

    # validation set
    val_set = ImpDataset(X=val['X'], imp_X=val['imp_X'], W=val['W'], args=args, scaler=train_set.scaler)
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            shuffle=False)

    test_set = ImpDataset(X=test['X'], imp_X=test['imp_X'], W=test['W'], args=args, scaler=train_set.scaler)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader
