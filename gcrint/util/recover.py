import torch


def extract_all_sub_series(dataset):
    seq_len = dataset.args.seq_len
    T, D = dataset.X.shape
    # uniformly cut the dataset to batch
    batch = []
    t_batch = []
    for idx in range(0, T, seq_len):
        t = idx
        if T - idx <= seq_len:
            t = T - seq_len - 1
        sample = dataset[t]
        batch.append(sample)
        t_batch.append(t)
    x_batch = torch.stack([sample['x'] for sample in batch])
    w_batch = torch.stack([sample['w'] for sample in batch])
    wo_batch = torch.stack([sample['wo'] for sample in batch])

    batch = {'t': t_batch, 'x': x_batch, 'w': w_batch, 'wo': wo_batch}
    return batch


def build_xhat(dataset, batch):
    # extract parameter
    T, D = dataset.X.shape
    seq_len = dataset.args.seq_len
    x_hat_batch = batch['xhat']
    t_batch = batch['t']

    # merge x_hat_batch into one tensor
    last_x_hat = x_hat_batch[-1, :, :]
    X_hat = x_hat_batch[:-1, :, :].reshape(-1, D)
    last_t = T - (t_batch[-2] + seq_len)
    X_hat = torch.cat([X_hat, last_x_hat[-last_t:]])

    # inverse transform X_hat to original scale
    X_hat = X_hat.cpu()
    X_hat = X_hat.data.numpy()
    X_hat = dataset.inverse_transform(X_hat)
    X_hat = dataset.np2torch(X_hat)
    return X_hat


def extract_all_sub_series_overlap(dataset):
    T, D = dataset.X.shape
    seq_len = dataset.args.seq_len

    batch = []
    t_batch = []
    for idx in range(T - seq_len):
        sample = dataset[idx]
        batch.append(sample)
        t_batch.append(idx)
    x_batch = torch.stack([sample['x'] for sample in batch])
    w_batch = torch.stack([sample['w'] for sample in batch])
    xi_batch = torch.stack([sample['xi'] for sample in batch])
    wi_batch = torch.stack([sample['wi'] for sample in batch])
    y_batch = torch.stack([sample['y'] for sample in batch])
    wy_batch = torch.stack([sample['wy'] for sample in batch])

    batch = {'t': t_batch, 'x': x_batch, 'w': w_batch, 'xi': xi_batch,
             'wi': wi_batch, 'y': y_batch, 'wy': wy_batch}

    return batch


def build_xhat_overlap(dataset, batch_xhat):
    T, D = dataset.X.shape
    seq_len = dataset.args.seq_len

    imp_X = torch.zeros(T, D, device=dataset.args.device)

    div = torch.zeros(T, 1, device=dataset.args.device)
    div[0] = 1.0
    div[-1] = 1.0
    ones = torch.ones(seq_len - 2, device=dataset.args.device, dtype=torch.float)

    for i in range(batch_xhat.shape[0]):
        imp_X[i + 1:i + seq_len - 1] += batch_xhat[i]
        div[i + 1:i + seq_len - 1, 0] += ones

    imp_X = imp_X.cpu()
    div = div.cpu()
    div[div == 0.0] = 1.0
    imp_X = imp_X / div

    imp_X = imp_X.data.numpy()
    imp_X = dataset.inverse_transform(imp_X)
    imp_X = dataset.np2torch(imp_X)

    imp_X[dataset.W == 1.0] = dataset.X[dataset.W == 1.0]
    imp_X[0] = dataset.X[0]
    imp_X[-1] = dataset.X[-1]

    return imp_X
