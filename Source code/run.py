import numpy as np
import torch
import random

import preprocess
import preprocess_mimic
import util

from hyperparams import all_hyperparams
from data_settings import all_settings


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    #random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    '''
    names: simulation, direct_{indep, dh, dep, no_h, full}, hierarch_{no_h, full}
    datasets: Synthetic, ADNI, MIMIC, SEER
    '''
    approach = 'hierarch_full'
    dataset_name = 'Synthetic' 
    data_settings = all_settings[dataset_name]
    if 'Synthetic' in dataset_name and 'high' in dataset_name and approach != 'simulation':
        data_settings['extra_feat'] = True
    
    if dataset_name != 'MIMIC':
        dataset = preprocess.get_and_bin_data(dataset_name, data_settings)
        data_packages = preprocess.split_data(dataset[0], dataset[1], dataset[2])
    else:
        dataset = preprocess_mimic.get_and_bin_data(dataset_name, data_settings)
        data_packages = preprocess_mimic.split_data(dataset[0], dataset[1], dataset[2], dataset[-1], data_settings)
    data_settings['min_time'], data_settings['max_time'] = dataset[3], dataset[4]
    
    train_data = data_packages[0]
    test_data = data_packages[1]
    val_data = data_packages[2]
    
    hyperparams = []
    if approach != 'simulation':
        hyperparams = all_hyperparams[dataset_name][approach]
        
    verbose = True
    mod = util.get_model_and_output(approach, train_data, test_data, val_data, data_settings, hyperparams, verbose)
    
    print(dataset_name + ',', approach, hyperparams)