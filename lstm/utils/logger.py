import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import os
import torch
from durbango import pickle_save


def display_stats(stats):
    print(
        'epoch={} -- rse={:0.4f} | mae={:0.4f} | mape={:0.4f} | mse={:0.4f} | rmse={:0.4f}'. \
            format(int(stats['epoch']),
                   stats['rse/infer'],
                   stats['mae/infer'],
                   stats['mape/infer'],
                   stats['mse/infer'],
                   stats['rmse/infer']))


class Logger:

    def __init__(self, args):
        log_dir = '../../../logs/icc/prediction/{}_{}_{}_{}'.format(args.model, args.dataset, args.sr, args.type)

        if args.tod:
            log_dir = log_dir + '_tod'
        if args.ma:
            log_dir = log_dir + '_ma'
        if args.mx:
            log_dir = log_dir + '_mx'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        args.log_dir = log_dir
        pickle_save(args, '{}/args.pkl'.format(args.log_dir))

        if args.verbose:
            print('[+] creating logger:', log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        self.args = args
        self.min_val_loss = np.inf
        self.patience = 0
        self.loss_fn = args.loss_fn
        self.best_model_save_path = os.path.join(args.log_dir, 'best_model.pth')

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
        description = 'train loss: {:.3f} val_loss: {:.3f} | best val_loss: {:.3f} patience: {}'.format(
            m.train_loss,
            m.val_loss,
            self.min_val_loss,
            self.patience)

        met_df.round(6).to_csv('{}/train_metrics.csv'.format(self.log_dir))

        if self.patience >= self.args.patience:
            self.stop = True
        return description

    def plot(self, x, y, yhat):
        plot_dir = os.path.join(self.log_dir, 'plots/')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        x = x.cpu().data.numpy()
        y = y.cpu().data.numpy()
        yhat = yhat.cpu().data.numpy()

        for flow in range(x.shape[1]):
            for idx in range(0, x.shape[0] - 501, 500):
                f, ax = plt.subplots(figsize=(11, 9))

                x_plt = x[idx, :, flow]
                y_plt = y[idx, :, flow]
                yhat_plt = yhat[idx, :, flow]

                gt = np.concatenate([x_plt, y_plt])
                pred = np.concatenate([x_plt, yhat_plt])

                plt.plot(pred, label='pred', color='red')
                plt.plot(gt, label='groud_truth', color='black')

                plt.title('Model: {} - Loss: {}'.format(self.args.model, self.args.loss_fn))
                plt.legend()
                plt.savefig(plot_dir + '/flow_{}_ts_{}.png'.format(flow, idx))
                plt.close()
