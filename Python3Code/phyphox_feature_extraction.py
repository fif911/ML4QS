#Extract features from data

import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction



df = pd.read_csv("Python3Code\phyphox-outputs\chapter2_result_250_labeled_aligned_min_extended_onehot_outlier.csv", skipinitialspace=True)

############################################# TEMPORAL ABSTRACTION ##########################################################

# numAbs = NumericalAbstraction()

# # Specify the values needed for the abstraction, window size, the fuction to use and the columns to aggregate
# windowSize = 10
# func = 'slope'
# cols = ['acc_x', 'acc_y', 'acc_z']

# # Perform abstraction to get abstracted data
# df_abs = numAbs.abstract_numerical(df, cols, windowSize, func)

# #Funtion which plots the specified column along with the abstracted column - make sure to use this function only when there is an abstracted column
# def plotAbstraction(df_abs, columns, window, func):
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
# plotAbstraction(df_abs, cols, windowSize, func)

############################################# FREQUENCY ABSTRACTION ##########################################################

# Sample frequency (Hz)
# fs = 10