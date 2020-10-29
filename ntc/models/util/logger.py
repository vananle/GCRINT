import os
import numpy as np
import pandas as pd
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter

from . import metric

class Logger:

    def __init__(self, args):
        log_dir = '../../result/{}_{}_{:0.1f}_{}'. \
            format(args.model, args.dataset, args.sr, args.type)
        if args.verbose:
            print('[+] creating logger:', log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        self.args = args
        self.validation_rse = None
        self.min_validation_rse = np.inf
        self.min_rse = np.inf
        self.patience = 0
        self.best_stats = None
        self.estimated_best_stats = None
        self.best_Xhat = None

    def summary(self, dataloader, Xhat, Xhat_scaled, loss, epoch):
        # extract data
        dataset  = dataloader.dataset
        X_scaled = dataset.X_scaled
        X        = dataset.X
        W        = dataset.W
        writer   = self.writer
        Xhat     = Xhat.reshape(X.shape)

        # report stats
        stats = {'loss': float(loss), 'epoch': int(epoch)}

        # report stats
        stats['sample_rse'], stats['infer_rse'] = metric.rse(X, Xhat, W)
        stats['sample_mae'], stats['infer_mae'] = metric.mae(X, Xhat, W)

        for key in stats:
            value = stats[key]
            writer.add_scalar(key, value, epoch)
        if stats['infer_rse'] < self.min_rse:
            self.min_rse = stats['infer_rse']
            self.best_stats = stats
            self.best_Xhat = Xhat
            self.dataloader = dataloader
        return stats

    def save(self):
        stats = self.best_stats
        for k in stats:
            stats[k] = [stats[k]]
        df = pd.DataFrame().from_dict(stats)
        # save metric
        path = os.path.join(self.log_dir, '{}.csv'.format(self.args.mode))
        df.to_csv(path)
        # save recovered data
        path = os.path.join(self.log_dir, '{}.mat'.format(self.args.mode))
        dataset  = self.dataloader.dataset
        T, D, D = dataset.X.shape
        D = D ** 2
        mdict = {'X': dataset.X.cpu().data.numpy().reshape(T, D),
                 'W': dataset.W.cpu().data.numpy().reshape(T, D),
                 'X_imp': self.best_Xhat.cpu().data.numpy().reshape(T, D)}
        savemat(path, mdict)

    def display(self):
        s = self.best_stats
        print('------------------------------------------------')
        print('Best result')
        print('epoch={} rse={:0.4f}/{:0.4f} mae={:0.4f}/{:0.4f}'.\
                format(int(s['epoch']),
                       s['sample_rse'], s['infer_rse'],
                       s['sample_mae'], s['infer_mae'],))

