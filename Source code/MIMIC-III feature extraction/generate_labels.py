"""
generate_labels.py
Original author: Shengpu Tang
Modified by Donna Tjandra

Generate labels for two adverse outcomes: ARF and shock.
"""

import pandas as pd
import numpy as np
import scipy.stats
import itertools
from collections import OrderedDict, Counter
from joblib import Parallel, delayed
from tqdm import tqdm as tqdm
import yaml
data_path = yaml.full_load(open('config.yaml'))['data_path']

import pathlib
pathlib.Path(data_path, 'labels').mkdir(parents=True, exist_ok=True)

examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['INTIME', 'OUTTIME']).sort_values(by='ICUSTAY_ID')
chartevents = pd.read_pickle(data_path + 'prep/chartevents.p')
procedures = pd.read_pickle(data_path + 'prep/procedureevents_mv.p')
inputevents = pd.read_pickle(data_path + 'prep/inputevents_mv.p')

ventilation = [
    '225792', # Invasive Ventilation
    '225794', # Non-invasive Ventilation
]

PEEP = [
    '220339', # PEEP set
]

vasopressors = [
    '221906', # Norepinephrine
    '221289', # Epinephrine
    '221662', # Dopamine
    '222315', # Vasopressin
    '221749', # Phenylephrine
]

## ARF: (PEEP) OR (mechanical ventilation), uncomment to generate labels
df_PEEP = chartevents[chartevents.ITEMID.isin(PEEP)].copy()
df_vent = procedures[procedures.ITEMID.isin(ventilation)].rename(columns={'t_start': 't'}).copy()
df_ARF = pd.concat([df_PEEP[['ICUSTAY_ID', 't']], df_vent[['ICUSTAY_ID', 't']]], axis=0)
df_ARF['ICUSTAY_ID'] = df_ARF['ICUSTAY_ID'].astype(int)
df_ARF = df_ARF.sort_values(by=['ICUSTAY_ID', 't']).drop_duplicates(['ICUSTAY_ID'], keep='first').reset_index(drop=True)
df_ARF = df_ARF.rename(columns={'t': 'ARF_ONSET_HOUR'})
df_ARF = pd.merge(examples[['ICUSTAY_ID']], df_ARF, on='ICUSTAY_ID', how='left')
df_ARF['ARF_LABEL'] = df_ARF['ARF_ONSET_HOUR'].notnull().astype(int)
print('ARF: ', dict(Counter(df_ARF['ARF_LABEL'])), 'N = {}'.format(len(df_ARF)), sep='\t')
df_ARF.to_csv(data_path + 'labels/ARF.csv', index=False)

## Shock: (one of vasopressors), uncomment to generate labels
df_vaso = inputevents[inputevents.ITEMID.isin(vasopressors)].rename(columns={'t_start': 't'}).copy()
df_shock = df_vaso.copy()
df_shock['ICUSTAY_ID'] = df_shock['ICUSTAY_ID'].astype(int)
df_shock = df_shock.sort_values(by=['ICUSTAY_ID', 't']).drop_duplicates(['ICUSTAY_ID'], keep='first').reset_index(drop=True)
df_shock = df_shock.rename(columns={'t': 'Shock_ONSET_HOUR'})
df_shock = pd.merge(examples[['ICUSTAY_ID']], df_shock, on='ICUSTAY_ID', how='left')
df_shock['Shock_LABEL'] = df_shock['Shock_ONSET_HOUR'].notnull().astype(int)
print('Shock: ', dict(Counter(df_shock['Shock_LABEL'])), 'N = {}'.format(len(df_shock)), sep='\t')
df_shock.to_csv(data_path + 'labels/Shock.csv', index=False)

##################################################################################################
'''
added labels for death
'''
patients = pd.read_csv('/data3/data/mimic3/' + 'PATIENTS.csv', parse_dates=['DOB', 'DOD'], usecols=['SUBJECT_ID', 'DOB', 'DOD'])
admissions = pd.read_csv('/data3/data/mimic3/' + 'ADMISSIONS.csv', parse_dates=['DEATHTIME'], usecols=['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'])

examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['INTIME', 'OUTTIME']).sort_values(by='ICUSTAY_ID') # Only Metavision

examples = pd.merge(examples, patients, on='SUBJECT_ID', how='left')
examples = pd.merge(examples, admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
examples['AGE'] = examples.apply(lambda x: (x['INTIME'] - x['DOB']).total_seconds(), axis=1) / 3600 / 24 / 365.25

examples['LOS'] = examples['LOS'] * 24

#generate labels
examples = examples_[(examples_.DEATHTIME >= examples_.INTIME + timedelta(hours=T)) | (examples_.DEATHTIME.isnull())]
death_lab = examples[['ICUSTAY_ID', 'DEATHTIME']]
death_lab['label'] = 1 - examples['DEATHTIME'].isnull()
death_times = np.array((death_lab['DEATHTIME'] - examples['INTIME']) / pd.Timedelta('1 hour'))
death_times[death_lab['label'] == 0] = ((examples['OUTTIME'] - examples['INTIME']) / pd.Timedelta('1 hour'))[death_lab['label'] == 0]
death_lab['DEATHTIME'] = ((examples['OUTTIME'] - examples['INTIME']) / pd.Timedelta('1 hour'))#death_times
death_lab.to_csv(data_path + 'labels/Death_with_cen_times.csv', index=False)

##################################################################################################
'''
map patients to visits, creates the pat_vis_map file used by the split function in preprocess_mimic
'''

#uncomment to generate a mapping from hospital visit id to patient id
pat_map = examples[['SUBJECT_ID', 'ICUSTAY_ID']]
print(np.unique(np.unique(np.array(pat_map['SUBJECT_ID']), return_counts=True)[1], return_counts=True))
pat_map.to_csv(data_path + 'pat_to_visit.csv', index=False)
pat_map['SUBJECT_ID'].to_csv(data_path + 'pop.csv', index=False)
