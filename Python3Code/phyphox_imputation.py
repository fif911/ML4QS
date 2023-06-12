##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Chapter3.ImputationMissingValues import ImputationMissingValues

# Set up the file names and locations.
DATA_PATH = Path('./phyphox-outputs/')    
DATASET_FNAME = 'chapter2_result_250_labeled_aligned_min_extended_onehot_outlier.csv'
RESULT_FNAME = 'imputation_results.csv'

dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
dataset.index = pd.to_datetime(dataset.index)

measurements = ['lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'loc_speed']
labels = ['label','label_cycling','label_hammocking','label_running','label_sitting','label_walking']

MisVal = ImputationMissingValues()
imputed_interpolation_dataset = pd.DataFrame()

for label in labels:
    imputed_interpolation_dataset[label] = dataset[label]

for measurement in measurements:
    imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), measurement)
    dataset[measurement] = imputed_interpolation_dataset[measurement]


RESULT_PATH = Path('./phyphox-outputs/')
dataset.to_csv(RESULT_PATH / RESULT_FNAME)