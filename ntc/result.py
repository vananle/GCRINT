import re
import os
import time
import argparse
import pandas as pd

def parse(args):

    table = []
    columns = ['sr', 'ter', 'ter_scaled', 'mae', 'mae_scaled']
    args.dataset = 'abilene_tm'
    args.type    = 'uniform'
    args.seed    = 1
    for sr in [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.4]:
        args.sr = sr
        if args.reduce_data:
            path = 'log/{}/{}_reduce/{}/{:0.3f}_{}.csv'.format(args.method, args.dataset, args.type, args.sr, args.seed)
        else:
            path = 'log/{}/{}/{}/{:0.3f}_{}.csv'.format(args.method, args.dataset, args.type, args.sr, args.seed)
        if os.path.exists(path):
            data = pd.read_csv(path)
            mae = re.findall(r'\d+\.\d+', str(data['mae'].iloc[0]))[0]
            mae = float(mae)
            mae_scaled = re.findall(r'\d+\.\d+', str(data['mae_scaled'].iloc[0]))[0]
            mae_scaled = float(mae_scaled)
            row = [args.sr, float(data['ter']), float(data['ter_scaled']), mae, mae_scaled]
        else:
            print('not exist', path)
            row = [args.sr, 0, 0, 0, 0]
        table.append(row)
    df = pd.DataFrame(table, columns=columns)
    return df

def libre_calc(df):
    df.to_csv('/tmp/result.csv')
    time.sleep(0.5)
    os.system('/opt/libreoffice6.3/program/soffice --calc /tmp/result.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ntc',
                        choices=['nocnn', 'ntc', 'cp'])
    parser.add_argument('--reduce_data', action='store_true')
    args   = parser.parse_args()
    df = parse(args)
    libre_calc(df)

if __name__ == '__main__':
    main()

