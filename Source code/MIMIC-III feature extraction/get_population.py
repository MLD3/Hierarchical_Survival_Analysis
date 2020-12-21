import pickle
import pandas as pd
import numpy as np

data_folder = ''#data_path from config.yaml

labels = pd.read_csv(data_folder + 'labels/ARF.csv') \
                   .rename(columns={'ICUSTAY_ID': 'ID'})

labels = pd.DataFrame(data=labels['ID'], columns=pd.Index(['ID'], name='cols'))
print(labels)
labels.to_csv(data_folder + 'pop.csv', index=False)
