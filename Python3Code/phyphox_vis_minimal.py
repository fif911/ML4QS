import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Set up the file names and locations.
DATA_PATH = Path('./phyphox-outputs/') 
# DATASET_FNAME = 'chapter2_result_250_labeled_aligned_min_extended_onehot_outlier.csv'
# DATASET_FNAME = 'lowpass_results.csv'
# DATASET_FNAME = 'kalman_filter_full_unlabeled_ds.csv'
DATASET_FNAME = 'lowpass_results.csv'
RESULT_PATH = Path('./phyphox-outputs/figures/phyphox_vis_minimal/')
RESULT_FNAME = 'lowpass_figure.csv'

labels = ['walking', 'running', 'cycling', 'sitting', 'hammocking']

# Load the CSV file into a pandas DataFrame
dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)

# dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')
dataset['time'] = pd.to_datetime(dataset.index, format='%Y-%m-%d %H:%M:%S')
dataset['formatted_date'] = dataset['time'].dt.strftime('%H:%M')
for label in labels:
    dataset[label + '_label'] = dataset['label'].apply(lambda x: 1 if x == label else 0)

fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True,sharey=False)

dataset.plot('formatted_date', y=['lin_acc_x'], ax=axes[0])
dataset.plot('formatted_date', y=['lin_acc_y'], ax=axes[1])
dataset.plot('formatted_date', y=['lin_acc_z'], ax=axes[2])
dataset.plot('formatted_date', y=['loc_speed'], ax=axes[3], linestyle='solid')
dataset.plot('formatted_date', y=[label + '_label' for label in labels], ax=axes[4], marker='o', linestyle='')

plt.savefig('./figures/lowpass.png')
plt.savefig('./figures/lowpass.pdf')

plt.show()

# slices_activities = {'hammocking': [500, 600], 'walking': [1500, 1600], 'running': [4500, 4600], 'sitting': [10500, 10600], 'cycling': [20000, 20100]}


# for activity in slices_activities:
#     df_subset = dataset.iloc[slices_activities[activity][0]:slices_activities[activity][1]]
#     fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True,sharey=False)
#     i = 0
#     for measurement in ['lin_acc', 'acc']:
#         for coordinate in ['_x', '_y', '_z']:
#             df_subset.plot('formatted_date', y=[measurement + coordinate ], ax=axes[i])
#             i+=1
    
#     plt.title(activity)
#     plt.savefig(f'./figures/raw_{activity}_extract.png')
#     plt.savefig(f'./figures/raw_{activity}_extract.pdf')
#     plt.show()

