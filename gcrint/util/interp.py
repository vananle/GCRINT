import os

import numpy as np
from joblib import delayed, Parallel
from scipy import interpolate


def interp1d(X, W, d):
    # extract time-series d
    T, D = X.shape
    x = X[:, d]
    w = W[:, d]

    # extract sampled data
    sample_indices = np.where(w)[0]
    sample_values = x[sample_indices]

    # fill sampled data with 0 at front and rear if needed
    mean = np.mean(sample_values)
    if 0 not in sample_indices:
        sample_indices = np.insert(sample_indices, 0, 0)
        sample_values = np.insert(sample_values, 0, mean)
    if T - 1 not in sample_indices:
        sample_indices = np.insert(sample_indices, -1, T - 1)
        sample_values = np.insert(sample_values, -1, mean)

    # interpolate
    f = interpolate.interp1d(sample_indices, sample_values)
    infer_indices = np.arange(T)
    infer_values = f(infer_indices)

    # save interpolated values
    return infer_values


def interp(X, W):
    T, D = X.shape
    list_x_interp = Parallel(n_jobs=os.cpu_count())(delayed(interp1d)(X, W, d) for d in range(D))
    X_interp = np.stack(list_x_interp).T
    return X_interp


def main():
    import time
    from metric import rse_np
    from data import load_matlab_matrix

    X = load_matlab_matrix('../../data/data/abilene_tm.mat')
    W = load_matlab_matrix('../../data/mask/abilene_tm/uniform/0.4_1.mat')
    tic = time.time()
    X_interp = interp(X, W)
    toc = time.time()

    sample_rse, infer_rse = rse_np(X, X_interp, W)
    print('[+] linear interpolation: sample_rse={:0.4f} infer_rse={:0.4f} time={:0.4f}'. \
          format(sample_rse, infer_rse, toc - tic))


if __name__ == '__main__':
    main()
