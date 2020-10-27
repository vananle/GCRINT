import argparse


def get_padding_conv1d_same(kernel_size):
    # assert kernel_size % 2 == 0
    return kernel_size // 2


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm'])
    parser.add_argument('--impset', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--savingpath', type=str, default='../../data/impdata')
    parser.add_argument('--sr', type=float, default=0.4)
    parser.add_argument('--type', type=str, default='uniform')
    parser.add_argument('--p_validation', type=float, default=0.3,
                        help='the proportion to leave observed data out for validation')

    # parameter for model
    parser.add_argument('--model', type=str, default='gcrint')
    parser.add_argument('--gcn_indim', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lstm_hidden', type=int, default=32)
    parser.add_argument('--residual_channels', type=int, default=64)
    parser.add_argument('--apt_size', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=64)

    parser.add_argument('--imp_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mse', 'mae', 'mse_u', 'mae_u'])
    parser.add_argument('--lamda', type=float, default=2.0)

    # parameter for routing_alg
    parser.add_argument('--routing_alg', type=str, default='sr',
                        choices=['sr', 'sp', 'or', 'ta'])
    parser.add_argument('--mon_policy', type=str, default='random',
                        choices=['heavy_hitter', 'fluctuation', 'fgg', 'random'])
    parser.add_argument('--te_step', type=int, default=1000)

    # parameter for optimizer
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')

    # verbosity
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run_te', action='store_true')
    parser.add_argument('--plot_results', action='store_true')

    # get args
    args = parser.parse_args()
    args.val_batch_size = args.imp_batch_size

    return args


def print_args(args):
    print('-------------------------------------')
    print('[+] Time-series recovering experiment')
    if args.test:
        print('|--- Run Test')
    elif args.run_te:
        print('|--- Do Traffic Engineering')
    else:
        print('|--- Run Train')

    print('-------------------------------------')
    print('    - datapath      :', args.datapath)
    print('    - dataset       :', args.dataset)
    print('    - train size    : {}x{}'.format(args.train_size, args.nSeries))
    print('    - val size      : {}x{}'.format(args.val_size, args.nSeries))
    print('    - test size     : {}x{}'.format(args.test_size, args.nSeries))
    print('    - type          :', args.type)
    print('    - seed          :', args.seed)
    print('    - sr            :', args.sr)
    print('    - p_validation  :', args.p_validation)
    print('-------------------------------------')
    if args.run_te:
        print('    - routing_alg        :', args.routing)
        print('    - mon_policy     :', args.mon_policy)
        print('    - te_step        :', args.te_step)
    print('-------------------------------------')
    print('    - model          :', args.model)
    print('    - num layers     :', args.num_layers)
    print('    - residual_channels     :', args.residual_channels)
    print('    - lstm hidden    :', args.lstm_hidden)
    print('    - dropout        :', args.dropout)
    print('    - lamda          :', args.lamda)
    print('-------------------------------------')
    print('    - device             :', args.device)
    print('    - imp_batch_size   :', args.imp_batch_size)
    print('    - val_batch_size     :', args.val_batch_size)
    print('    - test_batch_size    :', args.test_batch_size)
    print('    - seq_len            :', args.seq_len)
    print('    - loss_fn            :', args.loss_fn)
    print('    - num_epoch          :', args.num_epoch)
    print('    - learning_rate      :', args.learning_rate)
    print('    - patience           :', args.patience)
    print('    - log dir            :', args.log_dir)
    print('-------------------------------------')
