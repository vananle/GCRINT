import os

import models
import torch
import util
from tqdm import trange


def main():
    # Handle parameters
    args = util.get_args()

    # Select gpu
    device = torch.device(args.device)
    args.device = device

    # Load data
    train_loader, val_loader, test_loader = util.get_dataloader(args)

    train_dataloader, train_val_dataloader = train_loader
    val_dataloader, val_val_dataloader = val_loader
    test_dataloader, test_val_dataloader = test_loader

    args.train_size, args.nSeries = train_dataloader.dataset.X.shape
    args.val_size, args.val_nSeries = val_dataloader.dataset.X.shape
    args.test_size, args.test_nSeries = test_dataloader.dataset.X.shape

    # Create logger
    logger = util.Logger(args)

    # Display arguments
    util.print_args(args)

    # Create model
    model = models.get_model(args)

    # Create imputation engine

    engine = util.ImpEngine.from_args(model, scaler=None, args=args)

    # Training

    if args.impset == 'train':
        data_loader = train_dataloader
        val_loader = train_val_dataloader
    elif args.impset == 'val':
        data_loader = val_dataloader
        val_loader = val_val_dataloader
    elif args.impset == 'test':
        data_loader = test_dataloader
        val_loader = test_val_dataloader
    else:
        raise NotImplementedError

    if not args.test:
        iterator = trange(args.num_epoch)

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
                loss = engine.train(data_loader)
                engine.scheduler.step()
                with torch.no_grad():
                    # metrics = (val_loss, rse, mae, mape, mse, rmse)
                    Xhat_val, val_metrics = engine.validation(data_loader, val_loader)

                    m = dict(train_loss=loss, val_loss=val_metrics[0], val_rse=val_metrics[1],
                             val_mae=val_metrics[2], val_mape=val_metrics[3], val_mse=val_metrics[4],
                             val_rmse=val_metrics[5])

                # report stats
                description = logger.summary(m, engine.model)
                if logger.stop:
                    break
                description = 'Epoch: {} '.format(epoch) + description
                iterator.set_description(description)

        except KeyboardInterrupt:
            pass

    # data recovery
    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
    with torch.no_grad():
        # metrics = (rse, mae, mape, mse, rmse)
        imp_X, metrics, metrics_li = engine.imputation(data_loader)

        m = dict(imp_rse=metrics[0], imp_mae=metrics[1], imp_mape=metrics[2], imp_mse=metrics[3],
                 imp_rmse=metrics[4])
        # m_li = dict(imp_rse=metrics_li[0], imp_mae=metrics_li[1], imp_mape=metrics_li[2], imp_mse=metrics_li[3],
        #             imp_rmse=metrics_li[4])

        logger.imputation_summary(m=m, X=data_loader.dataset.X, imp_X=imp_X,
                                  W=data_loader.dataset.W, save_imp=True)


if __name__ == '__main__':
    main()
