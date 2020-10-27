import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='brits_abilene_tm',
                    choices=['brits_abilene_tm', 'brits_geant_tm', 'brits_brain_tm',
                             'crind_abilene_tm', 'crind_geant_tm', 'crind_brain_tm',
                             'gcrnn_abilene_tm', 'gcrnn_geant_tm', 'gcrnn_brain_tm'],
                    help='The abilene_sr is correspond to Abilene LL in the paper')
parser.add_argument('--impset', type=str, default='train',
                    choices=['train', 'val', 'test'])

parser.add_argument('--model', type=str, default='convbilstm', choices=['convbilstm', 'brits'])
parser.add_argument('--type', type=str, default='uniform', choices=['uniform', 'block'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--test', action='store_true')
parser.add_argument('--run_te', action='store_true')

args = parser.parse_args()

# number of random seed to test
repeat = 1

# number of sampling rate to test (sampling_rate = 1 - missing_rate)
srs = [0.1]

# repeat test for each random seed and sampling rate
# given the dataset, missing type, model in the command line argument

# TYPE = ['block']
# TYPE = ['uniform']
TYPE = ['uniform', 'block']
# DATASETS = ['brits_abilene_tm', 'brits_brain_tm']
DATASETS = ['gcrint_brain_tm', 'gcrint_abilene_tm']
# DATASETS = ['gcp_abilene_tm', 'gcp_brain_tm']
# DATASETS = ['ntc_abilene_tm', 'ntc_brain_tm']
# DATASETS = ['ntc_brain_tm']
# DATASETS = ['brits_abilene_tm', 'brits_brain_tm']

for data in DATASETS:
    for type in TYPE:
        for sr in srs:
            cmd = 'python train.py --device={} --dataset={} --type={} --seed={} --sr={}'. \
                format(args.device, data, type, 1, sr)

            if args.test:
                cmd = cmd + ' --test'
            if args.run_te:
                cmd = cmd + ' --run_te'
            print(cmd)
            os.system(cmd)
