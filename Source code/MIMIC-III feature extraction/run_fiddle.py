'''
script for automating FIDDLE preprocessing
'''

import extract_data
import generate_labels
import prepare_input
import make_features

import yaml
import argparse
import pandas as pd
import numpy as np

extract_data.main()
generate_labels.main()

data_path = yaml.full_load(open('config.yaml'))['data_path']
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default=data_path)
parser.add_argument('--population', type=str, default=data_path + 'pop.csv')
parser.add_argument('--T', type=float, default=5)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--theta_1', type=float, default=0.001)
parser.add_argument('--theta_2', type=float, default=0.001)
parser.add_argument('--theta_freq', type=float, default=1.0)
parser.add_argument('--stats_functions', nargs='+', default=['min', 'max', 'mean'])
args = parser.parse_args()

data_path = args.data_path
if not data_path.endswith('/'):
    data_path += '/'

population = args.population
T = int(args.T)
dt = args.dt
theta_1 = args.theta_1
theta_2 = args.theta_2
theta_freq = args.theta_freq
stats_functions = args.stats_functions

df_population = pd.read_csv(population).rename(columns={'ICUSTAY_ID': 'ID'}).set_index('ID')
N = len(df_population)
L = int(np.floor(T/dt))

prepare_input.main(20, 0.5)

if __name__ == '__main__':
    make_features.main(args)
