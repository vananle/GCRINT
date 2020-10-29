import torch
import numpy as np
from scipy.special import erf, erfinv

class NTCScaler:

    def fit_transform(self, X):
        '''Description of X
           - type (numpy ndarray)
           - shape (time x origin-destination)
           - example with abilene (48096 x 144)'''
        self.ub = np.max(X, axis=0)
        self.lb = np.min(X, axis=0)
        self.scale = self.ub - self.lb
        self.scale[self.scale==0] = 1
        self.eps = np.finfo(float).eps

        ub, lb, scale, eps = self.ub, self.lb, self.scale, self.eps

        # rescale to [0, 1]
        X = (X - lb) / scale

        # rescale to [-1, 1]
        X = 2 * X - 1

        # rescale to (-1, 1)
        X = np.clip(X, -1 + eps, 1 - eps)

        # erfinv
        X = erfinv(X)

        # rescale to [0, 1] again
        ub1 = np.max(X, axis=0)
        lb1 = np.min(X, axis=0)
        scale1 = ub1 - lb1
        scale1[scale1==0] = 1
        X = (X - lb1) / scale1

        self.ub1, self.lb1, self.scale1 = ub1, lb1, scale1

        return X

    def invert_transform(self, X):
        # output
        X = X.cpu().data.numpy()
        X = X.reshape(X.shape[0], -1)
        scale, lb, scale1, lb1 = self.scale, self.lb, self.scale1, self.lb1

        # rescale [0, 1] to erfinv output
        X = X * scale1 + lb1

        # erf
        X = erf(X)

        # rescale to [0, 1]
        X = (X + 1) / 2

        # rescale to input
        X = X * scale + lb

        # convert to torch
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.cuda()
        return X

