import os
import argparse

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abilene_tm_10k', choices=['abilene_tm_10k', 'brain_tm_10k', 'geant_tm_10k'])
    parser.add_argument('--model', type=str, default='ntc', choices=['nocnn', 'ntc', 'cp'])
    parser.add_argument('--sr_group', type=str, default='a', choices=['a', 'b'])
    args = parser.parse_args()

    # parameters
    srs = {'a': [0.1, 0.2, 0.3, 0.4, 0.5], 'b': [0.6, 0.7, 0.8, 0.9]}
    repeat = 1
    for sr in srs[args.sr_group]:
        for seed in range(1, repeat+1):
            for mode in ['train', 'val', 'test']:
                cmd = 'python3 main.py --dataset={} --model={} --sr={} --seed={} --mode={}'.format(
                        args.dataset, args.model, sr, seed, mode)
                print(cmd)
                os.system(cmd)

if __name__ == '__main__':
    main()
