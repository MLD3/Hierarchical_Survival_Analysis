from config import *
import pickle
import pandas as pd
import numpy as np
import time
import os

import argparse
from preprocessing import *
from run_fiddle import *

def main(args):
    if os.path.isfile(data_path + 'features/input_data.p'):
        input_fname = data_path + 'features/input_data.p'
        df_data = pd.read_pickle(input_fname)
    elif os.path.isfile(data_path + 'input_data.pickle'):
        input_fname = data_path + 'input_data.pickle'
        df_data = pd.read_pickle(input_fname)
    elif os.path.isfile(data_path + 'all_data.csv'):
        input_fname = data_path + 'all_data.csv'
        df_data = pd.read_csv(input_fname)

    # print('outcome = {}'.format(outcome))
    print('Input data file:', input_fname)
    print()
    print('Input arguments:')
    print('    {:<6} = {}'.format('T', T))
    print('    {:<6} = {}'.format('dt', dt))
    print('    {:<6} = {}'.format('\u03B8\u2081', theta_1))
    print('    {:<6} = {}'.format('\u03B8\u2082', theta_2))
    print('    {:<6} = {}'.format('\u03B8_freq', theta_freq))
    print('    {:<6} = {} {}'.format('k', len(stats_functions), stats_functions))
    print()
    print('N = {}'.format(N))
    print('L = {}'.format(L))
    print('', flush=True)

    print_header('1) Pre-filter')
    df_data = pre_filter(df_data, theta_1, args)

    print_header('2) Transform; 3) Post-filter')
    df_data, df_types = detect_variable_data_type(df_data, value_type_override, args)
    df_time_invariant, df_time_series = split_by_timestamp_type(df_data)

    # Process time-invariant data
    s, s_feature_names, s_feature_aliases = transform_time_invariant(df_time_invariant, args)

    # Process time-dependent data
    X, X_feature_names, X_feature_aliases = transform_time_dependent(df_time_series, args)
