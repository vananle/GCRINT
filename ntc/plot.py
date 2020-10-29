import os
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import models
from models import util

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm')
    parser.add_argument('--sr', type=float, default=0.11)
    parser.add_argument('--type', type=str, default='uniform')
    parser.add_argument('--model', type=str, default='nocnn',
                        choices=['nocnn', 'ntc', 'cp'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--reduce_data', action='store_true')
    parser.add_argument('--show_xhat', action='store_true')
    args = parser.parse_args()
    args.reduce_data = True

    args.time_length = 48096
    if args.reduce_data:
        args.time_length = 288
    args.dim = 12

    # load model
    if args.reduce_data:
        path = 'log/{}/{}_reduce/{}/{:0.3f}_{}.pkl'.format(args.model,
                    args.dataset, args.type, args.sr, args.seed)
    else:
        path = 'log/{}/{}/{}/{:0.3f}_{}.pkl'.format(args.model,
                    args.dataset, args.type, args.sr, args.seed)
    model = models.get_model(args)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))

    # load data
    dataloader, dataloader_full, args = util.get_dataloader(args)
    X = dataloader_full.dataset.X
    X_scaled = dataloader_full.dataset.X_scaled
    D = 144
    X = X.reshape(-1, D)
    X_scaled = X_scaled.reshape(-1, D)

    # load
    if torch.cuda.is_available() or args.show_xhat:
        with torch.no_grad():
            Xhat = []
            for batch in dataloader_full:
                xhat = model.forward(batch)
                Xhat.append(xhat)
            Xhat_scaled = torch.cat(Xhat).reshape(tuple(dataloader_full.dataset.X.shape))
            Xhat = dataloader_full.dataset.scaler.invert_transform(Xhat_scaled)
            Xhat = Xhat.reshape(-1, D)
            Xhat_scaled = Xhat_scaled.reshape(-1, D)

    # create folder
    path = 'plot'
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, args.model)
    if not os.path.exists(path):
        os.mkdir(path)
    if args.reduce_data:
        path = os.path.join(path, '{}_reduce'.format(args.dataset))
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        path = os.path.join(path, args.dataset)
        if not os.path.exists(path):
            os.mkdir(path)
    path = os.path.join(path, args.type)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, '{:0.3f}_{}'.format(args.sr, args.seed))
    if not os.path.exists(path):
        os.mkdir(path)
    folder = path

    # plot each flow
    for d in range(D):
        # plot in original scale
        plt.subplot(121)
        x = X[:, d]
        plt.plot(x[:288].cpu(), label='x')
        if torch.cuda.is_available() or args.show_xhat:
            xhat = Xhat[:, d]
            plt.plot(xhat[:288].cpu(), label='xhat')
        plt.legend()
        # plot in ntc scale
        plt.subplot(122)
        x_scaled = X_scaled[:, d]
        plt.plot(x_scaled[:288].cpu(), label='x_scaled')
        if torch.cuda.is_available() or args.show_xhat:
            xhat_scaled = Xhat_scaled[:, d]
            plt.plot(xhat_scaled[:288].cpu(), label='xhat_scaled')
        plt.legend()
        # save figure
        path = os.path.join(folder, 'flow_{}.png'.format(d))
        print('plotting', path)
        plt.savefig(path)
        plt.clf()

if __name__ == '__main__':
    main()

