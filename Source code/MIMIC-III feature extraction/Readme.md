# MIMIC-III Feature Extraction and Label Generation
The original FIDDLE code can be found here: https://gitlab.eecs.umich.edu/mld3/FIDDLE, and the original MIMIC-III extraction code can be found here: https://github.com/MLD3/FIDDLE-experiments.

The files used for this paper are included here for convenience.

## To Use FIDDLE

1. Run extract_data.py
2. Generate the labels by running generate_labels.py
3. Change the following files: config.yaml, get_population.py 
   (fill in the variables whose values are empty strings according to the instruction next to it)
4. To run the pipeline: python make_features.py --data_path [data_path from config.yaml] --population [where file with population ids is stored] --T 5 --dt 0.5 
