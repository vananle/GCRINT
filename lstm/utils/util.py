import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from scipy.sparse import linalg

DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler():
    def __init__(self):
        self.mins = None
        self.max = None
        self.nfeatures = 1

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.cuda()
        return X

    def fit(self, data):
        if len(data.shape) == 2:  # data in shape (time, nodes)
            self.mins = np.mean(data, axis=0)
            self.maxs = np.std(data, axis=0)
        elif len(data.shape) == 4:  # data in shape (sample, time, nodes, features)
            self.nsample, self.seq_len, self.nodes, self.nfeatures = data.shape
            self.mins, self.maxs = [], []
            self.mins_cuda, self.maxs_cuda = [], []
            for i in range(self.nfeatures):
                _min = np.min(np.reshape(data[..., i], newshape=[-1, self.nodes]), axis=0)
                _max = np.max(np.reshape(data[..., i], newshape=[-1, self.nodes]), axis=0)
                self.mins.append(_min)
                self.maxs.append(_max)

                self.mins_cuda.append(self.np2torch(_min))
                self.maxs_cuda.append(self.np2torch(_max))


        else:
            raise NotImplementedError('Data shape needs to be 2 or 4. Currently: {}'.format(data.shape))

    def transform(self, data):

        if self.mins is None:
            self.fit(data)

        if len(data.shape) == 2:  # data in shape (time, nodes)
            return (data - self.mins) / (self.maxs - self.mins)
        elif len(data.shape) == 4:  # data in shape (sample, time, nodes, features)
            self.nsample, self.seq_len, self.nodes, self.nfeatures = data.shape
            data = np.reshape(data, [-1, self.nodes, self.nfeatures])
            for i in range(self.nfeatures):
                data[..., i] = (data[..., i] - self.mins[i]) / (self.maxs[i] - self.mins[i])

            data = np.reshape(data, newshape=(self.nsample, self.seq_len, self.nodes, self.nfeatures))
            return data
        else:
            raise NotImplementedError('Data shape needs to be 2 or 4. Currently: {}'.format(data.shape))

    def inverse_transform(self, data):
        if self.mins is None:
            raise RuntimeError('Data was not transformed!')
        try:
            if len(data.shape) == 2:  # data in shape (time, nodes)
                return (data * (self.maxs - self.mins)) + self.mins
            elif len(data.shape) == 4:
                self.nsample, self.seq_len, self.nodes, self.nfeatures = data.shape
                data = np.reshape(data, [-1, self.nodes, self.nfeatures])
                for i in range(self.nfeatures):
                    data[..., i] = (data[..., i] * (self.maxs[i] - self.mins[i])) + self.mins[i]

                data = np.reshape(data, newshape=(self.nsample, self.seq_len, self.nodes, self.nfeatures))
                return data
            else:
                raise NotImplementedError('Data shape needs to be 2 or 4. Currently: {}'.format(data.shape))
        except TypeError:

            if len(data.shape) == 2:  # data in shape (time, nodes)
                return (data * self.maxs) + self.mins
            elif len(data.shape) == 4:

                self.nsample, self.seq_len, self.nodes, self.nfeatures = data.shape
                data = data.view(self.nsample * self.seq_len, self.nodes, self.nfeatures)
                for i in range(self.nfeatures):
                    data[..., i] = (data[..., i] * (self.maxs_cuda[i] - self.mins_cuda[i])) + self.mins_cuda[i]

                data = data.view(self.nsample, self.seq_len, self.nodes, self.nfeatures)
                return data
            else:
                raise NotImplementedError('Data shape needs to be 2 or 4. Currently: {}'.format(data.shape))


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def load_adj(pkl_filename, adjtype):
    if '.mat' in pkl_filename:
        adj_mx = loadmat(pkl_filename)['A'].astype(np.float)
        sensor_ids, sensor_id_to_ind = None, None
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    # savemat('adj_mx.mat', mdict={'A': adj_mx})

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)
    return mae, mape, rmse


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_tstep_metrics(model, device, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx)
            preds = scaler.inverse_transform(preds)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = yhat[..., i]
        # pred = scaler.inverse_transform(yhat[:, :, i])
        # pred = torch.clamp(pred, min=0., max=70.)
        pred = torch.clamp(pred, min=0., max=10e9)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat


def calc_tstep_metrics_missing(model, device, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, _, _) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx)
            preds = scaler.inverse_transform(preds)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = yhat[..., i]
        # pred = scaler.inverse_transform(yhat[:, :, i])
        # pred = torch.clamp(pred, min=0., max=70.)
        pred = torch.clamp(pred, min=0., max=10e9)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(['obs', 'sensor_id'])


def make_pred_df(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
                           yhat_last=_to_ser(scaler.inverse_transform(yhat[:, :, seq_length - 1])),
                           y_3=_to_ser(realy[:, :, 2]),
                           yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df


def make_graph_inputs(args, device):
    aptinit = None
    if not args.aptonly:
        sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        aptinit = None if args.randomadj else supports[0]  # ignored without do_graph_conv and add_apt_adj
    if args.aptonly:
        if not args.addaptadj and args.do_graph_conv: raise ValueError(
            'WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports
