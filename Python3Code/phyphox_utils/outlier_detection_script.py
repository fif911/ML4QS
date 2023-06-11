import copy

import numpy as np
import pandas as pd

dataset = pd.read_csv("../phyphox-outputs/chapter2_result_250_labeled_aligned_min_extended_onehot.csv",
                      skipinitialspace=True)
print(dataset.isnull().sum())

from Python3Code.util.VisualizeDataset import VisualizeDataset
from Python3Code.Chapter3.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection

# Create the outlier classes.
OutlierDistr = DistributionBasedOutlierDetection()
DataViz = VisualizeDataset("outlier_detection")
from typing import List

outlier_columns = ["lin_acc_x", "lin_acc_y", "lin_acc_z", "loc_speed"]
activities = dataset["label"].unique()
list_of_cleaned_datasets_by_activity: List = []
for activity in activities:
    dataset_by_activity = dataset[dataset["label"] == activity]
    dataset_by_activity = dataset_by_activity.reset_index(drop=True)

    for col in outlier_columns:
        print(f"Applying Chauvenet outlier criteria for activity {activity} column {col}")
        # TODO: skip for cycling loc_speed
        before_outlier_detection = copy.deepcopy(dataset_by_activity)
        dataset_by_activity = OutlierDistr.chauvenet(dataset_by_activity, col, 2)

        # DataViz.plot_binary_outliers(dataset_by_activity, col, col + '_outlier', title=activity)
        # set to NaN if outlier
        dataset_by_activity.loc[dataset_by_activity[col + '_outlier'] == True, col] = np.nan
        del dataset_by_activity[col + '_outlier']
    list_of_cleaned_datasets_by_activity.append(dataset_by_activity)

print(len(list_of_cleaned_datasets_by_activity))
concatened_dataset = pd.concat(list_of_cleaned_datasets_by_activity)
# Sort by time
concatened_dataset = concatened_dataset.sort_values(by=['time'])
# search for NaN
print(concatened_dataset.isnull().sum())
# save to csv
save_path = "../phyphox-outputs/chapter2_result_250_labeled_aligned_min_extended_onehot_outlier.csv"
concatened_dataset.to_csv(save_path,
                          index=False)
print(save_path)
