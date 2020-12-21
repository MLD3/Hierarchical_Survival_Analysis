'''
data preprocessing for mimic (binning time to events, split into train/test/validation)
'''

import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import get_data_mimic


#######################################################################################
'''
gets data and bins the time to events
'''
def get_and_bin_data(dataset, params):
    raw_data, event_times, labs, not_early = get_data_mimic.get_data_by_name(dataset, params) 
    
    return raw_data, event_times, labs, 0, params['num_bins'], not_early


'''
reformat labels so that label corresponds to a trajectory
'''
def get_trajectory_labels(labs):
    unique_labs = np.unique(labs, axis=0)
    new_labs = np.zeros((labs.shape[0],))
    
    for i in range(labs.shape[0]):
        for j in range(unique_labs.shape[0]):
            if np.all(unique_labs[j, :] == labs[i, :]):
                new_labs[i] = j
    
    return new_labs
   

'''
split data by patient (since mimic patients may have multiple visits)
'''
def split_data(raw_data, event_time, labs, not_early, params):
    pat_map_file = params['pat_vis_map']
    pat_map = pd.read_csv(pat_map_file).to_numpy()
    pat_map = pat_map[not_early, :]
    print('num unique pats', np.unique(pat_map[:, 0]).shape)
    traj_labs = get_trajectory_labels(labs)
    
    #split into training/test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    train_i, test_i = next(splitter.split(raw_data, traj_labs))
    
    train_pats = pat_map[train_i, 0]
    train_i = np.where(np.isin(pat_map[:, 0], train_pats))[0]
    test_i = np.setdiff1d(np.arange(raw_data.shape[0]), train_i)

    train_data = raw_data[train_i, :]
    train_labs = labs[train_i, :]
    train_event_time = event_time[train_i, :]
    
    pretest_data = raw_data[test_i, :]
    pretest_labs = labs[test_i, :]
    pretest_event_time = event_time[test_i, :]
    pretest_pats = pat_map[test_i, 0]
    
    #further split test set into test/validation
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    test_i, val_i = next(splitter.split(pretest_data, pretest_labs))
    test_pats = pretest_pats[test_i]
    test_i = np.where(np.isin(pretest_pats, test_pats))[0]
    val_i = np.setdiff1d(np.arange(pretest_pats.shape[0]), test_i)

    test_data = pretest_data[test_i, :]
    test_labs = pretest_labs[test_i, :]
    test_event_time = pretest_event_time[test_i, :]
    
    val_data = pretest_data[val_i, :]
    val_labs = pretest_labs[val_i, :]
    val_event_time = pretest_event_time[val_i, :]
    
    #package for convenience
    train_package = [train_data, train_event_time, train_labs]
    test_package = [test_data, test_event_time, test_labs]
    validation_package = [val_data, val_event_time, val_labs]

    return train_package, test_package, validation_package


'''
main block 
'''
if __name__ == '__main__':
    print(':)')