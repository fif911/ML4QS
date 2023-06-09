"""Label and align timestamps"""

import datetime

import pandas as pd

df = pd.read_csv("phyphox-outputs/chapter2_result_250_labeled.csv", skipinitialspace=True)

print("Shape before: ", df.shape)

df = df.dropna(thresh=2)
df.reset_index(drop=True, inplace=True)
print("Shape after: ", df.shape)
timestamp_string = df.iloc[0, 0]  # '2023-06-07 15:08:16.077333'
initial_time = datetime.datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S.%f")
# remove milliseconds
initial_time = initial_time.replace(microsecond=0)
print("Initial time: ", initial_time)

# Iterate over all rows and  increase the time by 250 ms for each row
for index, row in df.iterrows():
    if index == 0:
        df.at[index, 'Unnamed: 0'] = initial_time
    else:
        df.at[index, 'Unnamed: 0'] = points_add = initial_time + datetime.timedelta(milliseconds=250 * index)

df.to_csv("./phyphox-outputs/chapter2_result_250_labeled_aligned.csv", index=False)
print("Done!")
