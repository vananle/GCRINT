import torch
import itertools
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from .ntc_scaler import NTCScaler

class MissingDataset(Dataset):

    def __init__(self, args):
        # save parameter
        self.args = args

        # load data from matlab files
        path = '../../data/data/{}.mat'.format(args.dataset)
        X = load_matlab_matrix(path, 'X')
        path = '../../data/mask/{}/{}/{:0.1f}_{}.mat'.format(args.dataset,
                                                             args.type,
                                                             args.sr,
                                                             args.seed)
        W = load_matlab_matrix(path, 'W')

        # learn scaler
        self.scaler = NTCScaler()
        X_scaled = self.scaler.fit_transform(X)

        # extract data by mode
        X, X_scaled, W = self.extract_data(X, X_scaled, W)

        # convert data to pytorch
        self.X_scaled = np2torch(X_scaled)
        self.X = np2torch(X)
        self.W = np2torch(W)

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def extract_data(self, X, X_scaled, W):
        # extract parameters
        args = self.args
        T, D = X.shape
        D = int(np.sqrt(D)) # convert number of edge -> number of node
        num_train = int(T * args.p_train)
        num_val   = int(T * args.p_val)
        num_test  = int(T * args.p_test)

        # extract data by mode:
        if args.mode == 'train':
            start = 0
            end = num_train
        elif args.mode == 'val':
            start = num_train
            end = num_train + num_val
        elif args.mode == 'test':
            start = num_train + num_val
            end = num_train + num_val + num_test
        X = X.reshape(-1, D, D)[start:end, :, :]
        X_scaled = X_scaled.reshape(-1, D, D)[start:end, :, :]
        W = W.reshape(-1, D, D)[start:end, :, :]

        return X, X_scaled, W

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j, k = self.indices[idx]
        sample = {}
        sample['x'] = self.X_scaled[i, j, k]
        sample['i'] = i
        sample['j'] = j
        sample['k'] = k
        return sample

    def get_indices(self):
        indices = torch.where(self.W == 1) # todo
        indices = torch.stack(indices).T
        return indices

class FullDataset(Dataset):

    def __init__(self, args):
        # save parameter
        self.args = args

        # load data from matlab files
        path = '../../data/data/{}.mat'.format(args.dataset)
        X = load_matlab_matrix(path, 'X')
        path = '../../data/mask/{}/{}/{:0.1f}_{}.mat'.format(args.dataset,
                                                             args.type,
                                                             args.sr,
                                                             args.seed)
        W = load_matlab_matrix(path, 'W')

        # learn scaler
        self.scaler = NTCScaler()
        X_scaled = self.scaler.fit_transform(X)

        # extract data by mode
        X, X_scaled, W = self.extract_data(X, X_scaled, W)

        # convert data to pytorch
        self.X_scaled = np2torch(X_scaled)
        self.X = np2torch(X)
        self.W = np2torch(W)

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def extract_data(self, X, X_scaled, W):
        # extract parameters
        args = self.args
        T, D = X.shape
        D = int(np.sqrt(D)) # convert number of edge -> number of node
        num_train = int(T * args.p_train)
        num_val   = int(T * args.p_val)
        num_test  = int(T * args.p_test)

        # extract data by mode:
        if args.mode == 'train':
            start = 0
            end = num_train
        elif args.mode == 'val':
            start = num_train
            end = num_train + num_val
        elif args.mode == 'test':
            start = num_train + num_val
            end = num_train + num_val + num_test
        X = X.reshape(-1, D, D)[start:end, :, :]
        X_scaled = X_scaled.reshape(-1, D, D)[start:end, :, :]
        W = W.reshape(-1, D, D)[start:end, :, :]

        return X, X_scaled, W

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j, k = self.indices[idx]
        sample = {}
        sample['x'] = self.X_scaled[i, j, k]
        sample['i'] = i
        sample['j'] = j
        sample['k'] = k
        return sample

    def get_indices(self):
        return torch.Tensor(list(itertools.product(*[[j for j in range(self.X.shape[i])] for i in range(len(self.X.shape))]))).type(torch.long)

def load_matlab_matrix(path, variable_name):
    X = loadmat(path)[variable_name]
    return X

def get_dataloader(args):
    # Loading data
    dataset    = MissingDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset_full    = FullDataset(args)
    dataloader_full = DataLoader(dataset_full, batch_size=args.batch_size, shuffle=False)

    # Assign dataset information to args
    args.time_length, args.dim, _ = dataset.X.shape
    args.dataset_length = len(dataset)

    return dataloader, dataloader_full, args

def np2torch(X):
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        X = X.cuda()
    return X


