import argparse

def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--dataset', type=str, default='abilene_tm_10k')
    parser.add_argument('--type', type=str, default='uniform')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--sr', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--p_train', type=float, default=0.7)
    parser.add_argument('--p_val', type=float, default=0.1)

    # parameter for model
    parser.add_argument('--model', type=str, default='ntc',
                        choices=['nocnn', 'ntc', 'cp'])

    # parameter for optimizer
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--log_epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--regularization', type=float, default=1e-2)

    # verbosity
    parser.add_argument('--verbose', action='store_true')

    # get args
    args = parser.parse_args()

    # add p_test
    args.p_test = round(1 - args.p_train - args.p_val, 1)

    return args

def print_args(args):
    print('-------------------------------------')
    print('[+] Time-series recovering experiment')
    print('-------------------------------------')
    print('    - dataset       :', args.dataset)
    print('    - type          :', args.type)
    print('    - mode          :', args.mode)
    print('    - seed          :', args.seed)
    print('    - sr            :', args.sr)
    print('    - validation    : train={} val={} test={}'.format(args.p_train, args.p_val, args.p_test))
    print('    - size          : {}x{}x{}'.format(args.time_length, args.dim, args.dim))
    print('    - dataset_length: {}'.format(args.dataset_length))
    print('-------------------------------------')
    print('    - model         :', args.model)
    print('    - regularization:', args.regularization)
    print('-------------------------------------')
    print('    - gpu           :', args.gpu)
    print('    - batch_size    :', args.batch_size)
    print('    - num_epoch     :', args.num_epoch)
    print('    - learning_rate :', args.learning_rate)
    print('-------------------------------------')

