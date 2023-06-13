#Extract features from data

import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction

# Specify the values needed for the abstraction, window size, the fuction to use and the columns to aggregate
windowSize = 10
cols = ['lin_acc_x', 'loc_speed']

df = pd.read_csv("Python3Code\phyphox-outputs\lowpass_results.csv", skipinitialspace=True)

############################################# TEMPORAL ABSTRACTION ##########################################################

numAbs = NumericalAbstraction()

# Function used to aggregate the data 
func = 'mean'

# Perform abstraction to get abstracted data
df_abs = numAbs.abstract_numerical(df, cols, windowSize, func)

#Funtion which plots the specified column along with the abstracted column - make sure to use this function only when there is an abstracted column
def plotNumAbstraction(df_abs, columns, window, func):
    for col in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df_abs[col], label=col, color='blue')
        plt.plot(df.index, df_abs[col + '_temp_' + func + '_ws_' + str(window) ], label = col + ' ' + func, color='red')

        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title('Differences between ' + col + ' and its Rolling '  + func)
        plt.legend()
        plt.show()

# Plot values
# plotNumAbstraction(df_abs, cols, windowSize, func)

############################################# Classify Feature ###############################################################

cols = ['loc_speed']
col = 'loc_speed'

def checkLow(x, low):
    if x <= low:
        return 1
    else: 
        return 0
    
def checkMedium(x, low, medium):
    if x > low and x <= medium:
        return 1
    else: 
        return 0
    
def checkHigh(x, medium, high):
    if x > medium and x <= high:
        return 1
    else: 
        return 0


def labelFeatures(df, cols):

    #Get all maximum/minimum values from dataframe
    maximums = df_abs.max(axis='index')
    minimums = df_abs.min(axis='index')

    for col in cols:
        # Get the maximum and minimum value of the current column
        max = maximums[col]
        min = minimums[col]

        #Get the difference between the maximum and the minimum value for the column
        range = abs(max - min)
        unit = range/3

        # Create boundaries for the low, medium and high values
        low = min + unit
        medium = min + 2*unit
        high = min + 3*unit

        # df['label_' + col] = df[col].apply(lambda x: labelValue(x, low, medium, high))

        df['high_' + col] = df[col].apply(lambda x: checkLow(x, low))
        df['medium_' + col] = df[col].apply(lambda x: checkMedium(x,low, medium))
        df['low_' + col] = df[col].apply(lambda x: checkHigh(x, medium, high))

    return df

df = labelFeatures(df, cols)

print(df.columns)
print(df)

# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df_abs[col], label=col, color='blue')
# plt.plot(df.index, df_abs['label_' + col], label=col, color='red')

# plt.xlabel('Index')
# plt.ylabel(col)
# plt.title('Differences between ' + col + ' and its Rolling '  + func)
# plt.legend()
# plt.show()



############################################# FREQUENCY ABSTRACTION ##########################################################

# # Sample frequency (Hz)
# fs = 10

# FreqAbs = FourierTransformation()

# df_freq = FreqAbs.abstract_frequency(copy.deepcopy(df), cols, windowSize, fs)

# def plotFreqAbstraction(df_freq, columns):
#     for col in columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index, df_freq[col], label=col, color='blue')
#         plt.plot(df.index, df_freq[col + '_max_freq'], label = col + ' max frequency', color='red')
#         plt.xlabel('Index')
#         plt.ylabel(col)
#         plt.title('Frequency changes in ' + col)
#         plt.legend()
#         plt.show()

# ###########################################
#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index, df_freq[col], label=col, color='blue')
#         plt.plot(df.index, df_freq[col + '_freq_weighted'], label = col + ' weighted frequency', color='yellow')

#         plt.xlabel('Index')
#         plt.ylabel(col)
#         plt.title('Frequency changes in ' + col)
#         plt.legend()
#         plt.show()

# #############################################
#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index, df_freq[col], label=col, color='blue')
#         plt.plot(df.index, df_freq[col + '_pse'], label = col + ' pse', color='green')

#         plt.xlabel('Index')
#         plt.ylabel(col)
#         plt.title('Frequency changes in ' + col)
#         plt.legend()
#         plt.show()

# window_overlap = 0.9
#         skip_points = int((1-window_overlap) * ws)
#         dataset = dataset.iloc[::skip_points,:]


# plotFreqAbstraction(df_freq, cols)



# frequencies = []
# values = []
# for col in df_freq.columns:
#     val = re.findall(r'freq_\d+\.\d+_Hz', col)
#     if len(val) > 0:
#         frequency = float((val[0])[5:len(val)-4])
#         frequencies.append(frequency)
#         values.append(df_freq.loc[df_freq.index, col])


