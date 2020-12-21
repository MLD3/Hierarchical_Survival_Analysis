'''
gets mimic data for downstream processing 
'''

import numpy as np
import pandas as pd
import sparse 
import copy


'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[1:]
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    f.close()
    return c


'''
normalize data to 0-1 range
'''
def normalize(data):
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    
    dims = data.shape
    mins = np.tile(mins, (dims[0], 1))
    maxs = np.tile(maxs, (dims[0], 1))
    
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    data[:, :] = (data[:,:] - mins[:, :]) / (ranges)
    return data


'''
load time invariant data (FIDDLE output)
'''
def load_time_invar(data_source, data_root):
    raw_data = np.load(data_root + data_source + '/s.npz')
    data = sparse.COO(raw_data['coords'], raw_data['data'], tuple(raw_data['shape']))
    data = data.todense()

    pats = get_file(data_root + 'pop.csv', 1)[1:]
    pats = pats.reshape(-1)
    pat_sort_i = np.argsort(pats).reshape(-1)

    data = data[pat_sort_i, :]

    return data


'''
load time variant data (FIDDLE output)
'''
def load_time_var(data_source, data_root):
    raw_data = np.load(data_root + data_source + '/X.npz')
    data = sparse.COO(raw_data['coords'], raw_data['data'], tuple(raw_data['shape']))
    data = np.array(data.todense())

    num_segs = 1 #number of segments per feature vector
    seg_size = 1 #number of bins per segment
    ret_data = np.zeros((data.shape[0], 0))
    for i in range(num_segs):
        segment = np.sum(data[:, i*seg_size:(i*seg_size) + seg_size, :], axis=1)
        ret_data = np.concatenate((ret_data, segment), axis=1)

    ret_data = normalize(ret_data)
    return ret_data


'''
combine time in/variant data
'''
def combine_data(data_source, data_root):
    time_invar = load_time_invar(data_source, data_root)
    time_var = load_time_var(data_source, data_root)

    all_data = np.concatenate((time_invar, time_var), axis=1) 

    return all_data


'''
gets patient labels
first column is patient id
second columns is the time from alignment to conversion (if they convert, otherwise -1)
third column is label
fourth column indicates if patient is censored
'''
def get_labels(horizon, data_root):
    death_labs_file = data_root + 'labels/Death_with_cen_times.csv' 
    arf_labs_file = data_root + 'labels/ARF.csv'
    shock_labs_file = data_root + 'labels/Shock.csv' 

    death_raw = pd.read_csv(death_labs_file).to_numpy()
    arf_raw = pd.read_csv(arf_labs_file).to_numpy()
    shock_raw = pd.read_csv(shock_labs_file).to_numpy()

    raw_labs = [arf_raw, shock_raw, death_raw]
    event_times = 0 * np.ones((death_raw.shape[0], 3))
    final_labs = 0 * np.ones((death_raw.shape[0], 3))

    for i in range(len(raw_labs)):
        labs = raw_labs[i]
        had_event =  np.where(labs[:, -1].astype(int) == 1)[0]
        no_event =  np.where(labs[:, -1].astype(int) == 0)[0]
        final_labs[had_event, i] = 1
        event_times[had_event, i] = labs[had_event, 1].astype(float)
        event_times[no_event, i] = death_raw[no_event, 1].astype(float) - 0.001

        event_times[:, i][event_times[:, i] >= horizon] = horizon - 0.0001
        final_labs[:, i][event_times[:, i] >= horizon] = 0

    return final_labs, event_times


'''
gets mimic data and formats it for preprocessing
'''
def get_mimic(data_root, feat_folder):
    horizon = 12
    offset = 6 

    #get raw data
    raw_data = combine_data(feat_folder, data_root)
    raw_data = normalize(raw_data)
    labs, event_times = get_labels(horizon + offset, data_root)    
  
    early = np.where(event_times < 0.5)[0]
    if offset > 0:
        early = np.where(event_times < offset)[0]
    not_early = np.setdiff1d(np.arange(raw_data.shape[0]), early)
    event_times = event_times[not_early, :]
    labs = labs[not_early, :]
    raw_data = raw_data[not_early, :]
    event_times = np.floor(event_times) - offset

    var_tol = 2
    col_var = np.sum(raw_data, axis=0)
    low_var1 = np.where(np.logical_or(col_var < var_tol, col_var > raw_data.shape[0] - 2))[0]
    raw_data = np.delete(raw_data, low_var1, axis=1)

    event_order = np.argsort(event_times, axis=1)
    for i in range(event_order.shape[0]):
        for j in range(event_order.shape[1]):
            event = event_order[i, j]
            if labs[i, event] == 0:
                event_order[i, j] = -1
                
    print('event sequence distribution: ', np.unique(event_order, return_counts=True, axis=0))
    print('time to event distribution: ', np.unique(event_times, return_counts=True))
    print('label distribution: ', np.unique(labs, axis=0, return_counts=True))
    print('num data, num features: ', raw_data.shape)
    print('horiz cen', np.where(np.logical_and(np.sum(labs, axis=1)==0, np.sum(event_times, axis=1)==33))[0].shape)
    return raw_data, event_times.astype(int), labs.astype(int), not_early


'''
given the dataset name, calls the appropriate get_data or make_synthetic function
returns data in format: covariates, time to all events, labels 
'''
def get_data_by_name(dataset_name, params):
    print(dataset_name)
    
    if dataset_name == 'MIMIC':
        return get_mimic(params['location'], params['features'])


'''
main block
'''
if __name__ == '__main__':
    print(':)')