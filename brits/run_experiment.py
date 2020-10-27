import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='abilene_tm',
                    choices=['abilene_tm', 'geant_tm', 'brain_tm'],
                    help='The abilene_sr is correspond to Abilene LL in the paper')
parser.add_argument('--impset', type=str, default='train',
                    choices=['train', 'val', 'test'])

parser.add_argument('--model', type=str, default='convbilstm', choices=['convbilstm', 'brits'])
parser.add_argument('--type', type=str, default='uniform', choices=['uniform', 'block'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()

# hyperparameters
num_epoch = 200
num_hidden = 32

# number of random seed to test
repeat = 1

# number of sampling rate to test (sampling_rate = 1 - missing_rate)
srs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# repeat test for each random seed and sampling rate
# given the dataset, missing type, model in the command line argument

# TYPE = ['block']
TYPE = ['uniform', 'block']
DATASETS = ['abilene_tm', 'geant_tm', 'brain_tm']

for type in TYPE:
    for sr in srs:
        cmd = 'python main.py --device={} --dataset={} --type={} --seed={} --sr={} --imp_batch_size 64 --impset {}'. \
            format(args.device, args.dataset, type, 1, sr, args.impset)

        if args.test:
            cmd = cmd + ' --test'
        print(cmd)
        os.system(cmd)
