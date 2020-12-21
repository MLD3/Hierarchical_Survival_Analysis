# Readme

## To Use FIDDLE

1. Change the following files: config.yaml, get_population.py 
   (fill in the variables whose values are empty strings according to the instruction next to it)

2. To run the pipeline: python make_features.py --data_path [data_path from config.yaml] --population [where file with population ids is stored] --T 5 --dt 0.5 

For more documentation of FIDDLE, see: https://gitlab.eecs.umich.edu/mld3/FIDDLE 

## After Using FIDDLE

Generate the labels by running generate_labels.py