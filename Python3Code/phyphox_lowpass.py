import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter

# Set up the file names and locations.
DATA_PATH = Path('./phyphox-outputs/')    
# DATASET_FNAME = 'chapter2_result_250_labeled_aligned_minimal.csv'
# DATASET_FNAME = 'kalman_filter_dataset.csv'
DATASET_FNAME = 'imputation_results.csv'
RESULT_FNAME = 'lowpass_results.csv'

dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
dataset.index = pd.to_datetime(dataset.index)

# We'll create an instance of our visualization class to plot the results.
DataViz = VisualizeDataset(__file__)

# And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
# speed has been removed because it gave negative values
periodic_measurements = ['lin_acc_x', 'lin_acc_y', 'lin_acc_z']


# Let us apply a lowpass filter and reduce the importance of the data above 2 Hz
LowPass = LowPassFilter()

# Determine the sampling frequency.
# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = 250
fs = float(1000)/milliseconds_per_instance
cutoff = 1.5 # 2 does not work because 2/cutoff has to be strickly between 0 and 1

# lowpass_dataset = pd.DataFrame()
# lowpass_dataset['label'] = dataset['label']
# # lowpass_dataset['time'] = dataset['time']
# lowpass_dataset['loc_speed'] = dataset['loc_speed']
# lowpass_dataset.index = dataset.index

for col in periodic_measurements:
    dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=20) # order is 5 because other the filter filter too many values
    dataset[col] = dataset[col + '_lowpass']

print(dataset)

# Store the final outcome.
dataset.to_csv(DATA_PATH / RESULT_FNAME)

# DataViz.plot_dataset(minimal_lowpass_dataset, ['acc_', 'loc_speed_', 'label'],
#                              ['like', 'like', 'like'],
#                              ['line', 'line', 'points'])
