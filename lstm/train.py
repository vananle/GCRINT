import os
import sys
import time

sys.path.append('..')
from routing import *

import models
import numpy as np
import torch
import utils
from tqdm import trange


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if 'abilene_tm' in args.dataset:
        args.nNodes = 12
        args.day_size = 288
    elif 'geant_tm' in args.dataset:
        args.nNodes = 22
        args.day_size = 96
    elif 'brain_tm' in args.dataset:
        args.nNodes = 9
        args.day_size = 1440
    elif 'sinet' in args.dataset:
        args.nNodes = 73
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    train_loader, val_loader, test_loader = utils.get_dataloader(args)

    args.train_size, args.nSeries = train_loader.dataset.X.shape
    args.val_size = val_loader.dataset.X.shape[0]
    args.test_size = test_loader.dataset.X.shape[0]

    in_dim = 2
    if args.tod:
        in_dim += 1
    if args.ma:
        in_dim += 1
    if args.mx:
        in_dim += 1

    args.in_dim = in_dim

    model = models.get_model(args)
    logger = utils.Logger(args)

    engine = utils.Trainer.from_args(model, train_loader.dataset.scaler, args)

    utils.print_args(args)

    if not args.test:
        iterator = trange(args.epochs)

        try:
            if os.path.isfile(logger.best_model_save_path):
                print('Model checkpoint exist!')
                print('Load model checkpoint? (y/n)')
                _in = input()
                if _in == 'y' or _in == 'yes':
                    print('Loading model...')
                    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
                else:
                    print('Training new model')

            for epoch in iterator:
                train_loss, train_rse, train_mae, train_mse, train_mape, train_rmse = [], [], [], [], [], []
                for iter, batch in enumerate(train_loader):

                    x = batch['x']  # [b, seq_x, n, f]
                    y = batch['y']  # [b, seq_y, n]

                    if y.max() == 0: continue
                    loss, rse, mae, mse, mape, rmse = engine.train(x, y)
                    train_loss.append(loss)
                    train_rse.append(rse)
                    train_mae.append(mae)
                    train_mse.append(mse)
                    train_mape.append(mape)
                    train_rmse.append(rmse)

                engine.scheduler.step()
                with torch.no_grad():
                    val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = engine.eval(val_loader)
                m = dict(train_loss=np.mean(train_loss), train_rse=np.mean(train_rse),
                         train_mae=np.mean(train_mae), train_mse=np.mean(train_mse),
                         train_mape=np.mean(train_mape), train_rmse=np.mean(train_rmse),
                         val_loss=np.mean(val_loss), val_rse=np.mean(val_rse),
                         val_mae=np.mean(val_mae), val_mse=np.mean(val_mse),
                         val_mape=np.mean(val_mape), val_rmse=np.mean(val_rmse))

                description = logger.summary(m, engine.model)

                if logger.stop:
                    break

                description = 'Epoch: {} '.format(epoch) + description
                iterator.set_description(description)
        except KeyboardInterrupt:
            pass

    # Metrics on test data
    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
    with torch.no_grad():
        test_met_df, x_gt, y_gt, y_real, yhat = engine.test(test_loader, engine.model, args.out_seq_len)
        test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics.csv'))
        print('Prediction Accuracy:')
        print(utils.summary(logger.log_dir))

    if args.plot:
        logger.plot(x_gt, y_real, yhat)

    x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
    y_gt = y_gt.cpu().data.numpy()
    yhat = yhat.cpu().data.numpy()

    # run TE
    if args.run_te:
        run_te(x_gt, y_gt, yhat, args)


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
