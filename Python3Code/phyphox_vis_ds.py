from util.VisualizeDataset import VisualizeDataset

# raad dataset from /Users/zakotianskyi/PycharmProjects/ML4QS/Python3Code
import pandas as pd
from pathlib import Path

dataset = pd.read_csv("./phyphox-outputs/chapter2_result_250.csv", skipinitialspace=True)

# Plot the data
DataViz = VisualizeDataset(__file__)

# Boxplot
# DataViz.plot_dataset_boxplot(dataset,
#                              ['acc_x', 'acc_y', 'acc_z'])

# # Plot all data
# DataViz.plot_dataset(dataset,
#                      # ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
#                      # ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
#                      # ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points']
#                      )

DataViz.plot_dataset(dataset,
                     ['acc', 'lin_acc', 'loc_speed', 'gyr', 'loc_L', 'light_lux', 'mang'],
                     ['like', 'like', 'exact', 'like', 'like', 'exact', 'like'],
                     ['line', 'line', 'line', 'line', 'like', 'line', 'line']
                     # ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
                     # ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                     # ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points']
                     )

print("done")
exit(0)