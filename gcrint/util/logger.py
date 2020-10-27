import os

import numpy as np
import pandas as pd
import torch
from durbango import pickle_save
from scipy.io import savemat


class Logger:

    def __init__(self, args):
        log_dir = '../../logs/icc/imputation/{}_{}_{}/{}_{}/{}_{}_{}/'. \
            format(args.model, args.num_layers, args.residual_channels, args.dataset, args.seq_len, args.type,
                   args.sr, args.seed)
        if args.verbose:
            print('[+] creating logger:', log_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        args.log_dir = log_dir
        pickle_save(args, '{}/args.pkl'.format(args.log_dir))

        self.log_dir = log_dir

        self.imp_saving_path = os.path.join(args.savingpath,
                                            '{}_{}_{}_{}'.format(args.model, args.dataset, args.sr, args.type))
        if not os.path.exists(self.imp_saving_path):
            os.makedirs(self.imp_saving_path)

        self.args = args
        self.min_val_loss = np.inf
        self.patience = 0
        self.best_model_save_path = os.path.join(args.log_dir, 'best_model_{}.pth'.format(args.impset))

        self.metrics = []
        self.stop = False
        self.Xhat = None

    def summary(self, m, model):
        m = pd.Series(m)
        self.metrics.append(m)
        if m.val_loss < self.min_val_loss:
            torch.save(model.state_dict(), self.best_model_save_path)
            self.patience = 0
            self.min_val_loss = m.val_loss
        else:
            self.patience += 1
        met_df = pd.DataFrame(self.metrics)
        description = 'train loss: {:.6f} val_loss: {:.6f} | best val_loss: {:.6f} patience: {}'.format(
            m.train_loss,
            m.val_loss,
            self.min_val_loss,
            self.patience)

        met_df.round(6).to_csv('{}/train_metrics_{}.csv'.format(self.log_dir, self.args.impset))

        if self.patience >= self.args.patience:
            self.stop = True
        return description

    def imputation_summary(self, m, X, imp_X, W, save_imp):
        print('Results: ', self.args.impset)
        print(m)

        m = pd.Series(m)
        met_df = pd.DataFrame([m])
        met_df.round(6).to_csv('{}/imp_metrics_{}.csv'.format(self.log_dir, self.args.impset))
        if save_imp:
            self.save_imp_data(X, W, imp_X)

    def save_imp_data(self, X, imp_X, W):
        X = X.cpu().data.numpy()
        imp_X = imp_X.cpu().data.numpy()
        W = W.cpu().data.numpy()
        savemat('{}/{}.mat'.format(self.imp_saving_path, self.args.impset), {'X': X, 'X_imp': imp_X, 'W': W})
