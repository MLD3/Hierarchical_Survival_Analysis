# Main Code

## Data Retrieval and Preprocessing

get_data.py: Constructs the synthetic data and retrieves ADNI and SEER  
get_data_mimic.py: Retrieves MIMIC-III data  
MIMIC-III feature extraction (directory): code for extracting features from MIMIC-III (includes FIDDLE code)  

preprocess.py: Splits the synthetic, ADNI, and SEER data into a training/test/validation set   
preprocess_mimic.py: Splits the MIMIC-III data into a training/test/validation set   

## Models and Training

direct.py: Specifies the model that predicts the target task directly  
hierarch.py: Specifies the model that uses hierarchical predictions  
simulation.py: Produces ground truth survival curves for the synthetic data  
util.py: Collection of helper functions used for training and evaluation  
run.py: Wrapper that calls all the necessary functions to train a model  

## Settings and Hyperparameters

hyperparams.py: Collection of hyperparameters used in the paper for each approach  
data_settings.py: Dataset-specific settings required for model training and evaluation  

## Instructions

1. Fill in the empty strings in data_settings with where the raw data is stored (more details in the file)
2. Change lines 27 and 28 in run.py to the approach and dataset desired, respectively (see lines 24 and 25 for more details)
3. Run run.py (e.g., python run.py)