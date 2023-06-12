#Extract features from data

import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction

# Specify the values needed for the abstraction, window size, the fuction to use and the columns to aggregate
windowSize = 10
cols = ['acc_x', 'acc_y', 'acc_z']

df = pd.read_csv("Python3Code\phyphox-outputs\chapter2_result_250_labeled_aligned_min_extended_onehot_outlier.csv", skipinitialspace=True)

############################################# TEMPORAL ABSTRACTION ##########################################################

# numAbs = NumericalAbstraction()

# Function used to aggregate the data 
# func = 'mean'

# # Perform abstraction to get abstracted data
# df_abs = numAbs.abstract_numerical(df, cols, windowSize, func)

# #Funtion which plots the specified column along with the abstracted column - make sure to use this function only when there is an abstracted column
# def plotNumAbstraction(df_abs, columns, window, func):
#     for col in columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index, df_abs[col], label=col, color='blue')
#         plt.plot(df.index, df_abs[col + '_temp_' + func + '_ws_' + str(window) ], label = col + ' ' + func, color='red')

#         plt.xlabel('Index')
#         plt.ylabel(col)
#         plt.title('Differences between ' + col + ' and its Rolling '  + func)
#         plt.legend()
#         plt.show()

# # Plot values
# plotNumAbstraction(df_abs, cols, windowSize, func)

############################################# FREQUENCY ABSTRACTION ##########################################################

# Sample frequency (Hz)
fs = 10

FreqAbs = FourierTransformation()

df_freq = FreqAbs.abstract_frequency(copy.deepcopy(df), cols, windowSize, fs)

def plotFreqAbstraction(df_freq, columns):
    for col in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df_freq[col], label=col, color='blue')
        plt.plot(df.index, df_freq[col + '_max_freq'], label = col + ' max frequency', color='red')
        plt.plot(df.index, df_freq[col + '_freq_weighted'], label = col + ' weighted frequency', color='yellow')
        plt.plot(df.index, df_freq[col + '_pse'], label = col + ' pse', color='green')

        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title('Frequency changes in ' + col)
        plt.legend()
        plt.show()

plotFreqAbstraction(df_freq, cols)



# frequencies = []
# values = []
# for col in df_freq.columns:
#     val = re.findall(r'freq_\d+\.\d+_Hz', col)
#     if len(val) > 0:
#         frequency = float((val[0])[5:len(val)-4])
#         frequencies.append(frequency)
#         values.append(df_freq.loc[df_freq.index, col])


