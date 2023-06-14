#Extract features from data

from pathlib import Path
import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction

# Overlap the windows of a dataframe
def windowOverlap(df, windowSize):
     # Specify the window overlap percentage
    window_overlap = 0.9
    # Calculate the number of points to skip between windows
    skip_points = int((1 - window_overlap) * windowSize)
    # Apply window overlap by selecting rows with a stride of skip_points
    df = df.iloc[::skip_points, :]

    return df



#Funtion which plots the specified column along with the abstracted column - make sure to use this function only when there is an abstracted column
def plotNumAbstraction(df_abs, columns, window, func):
    for col in columns:
        #Define file to save to
        RESULT_PATH = Path('./Python3Code/figures/phyphox_temporal_features/')
        RESULT_FNAME_PNG = f'temporal_features_' + col + '_' + func + '.png'
        RESULT_FNAME_PDF = f'temporal_features_' + col + '_' + func + '.pdf'

        plt.figure(figsize=(10, 6))
        plt.plot(df_abs.index, df_abs[col], label=col, color='blue')
        plt.plot(df_abs.index, df_abs[col + '_temp_' + func + '_ws_' + str(window) ], label = col + ' ' + func, color='red')

        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title('Differences between ' + col + ' and its Rolling '  + func)
        plt.legend()

        plt.savefig(RESULT_PATH / RESULT_FNAME_PNG)
        plt.savefig(RESULT_PATH / RESULT_FNAME_PDF)

        # plt.show()

def plotFreqAbstraction(df_freq, columns):

    # Obtain frequencies and corresponding values 
    # frequencies = []
    # values = []
    # for col in df_freq.columns:
    #     val = re.findall(r'freq_\d+\.\d+_Hz', col)
    #     if len(val) > 0:
    #         frequency = float((val[0])[5:len(val)-4])
    #         frequencies.append(frequency)
    #         values.append(df_freq.loc[df_freq.index, col])

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # plt.xlim([0, 5])
    # ax1.plot(frequencies, values, 'b+')
    # ax1.set_xlabel('Frequency (Hz)')
    # ax1.set_ylabel('$a$')
    # plt.show()

    for col in columns:

        RESULT_PATH = Path('./Python3Code/figures/phyphox_frequency_features/')
        RESULT_FNAME_PNG = f'frequency_features_' + col + '.png'
        RESULT_FNAME_PDF = f'frequency_features_' + col + '.pdf'

        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df_freq[col], label=col, color='blue')
        plt.plot(df.index, df_freq[col + '_max_freq'], label = col + ' max frequency', color='red')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title('Max Frequency ' + col)

        plt.subplot(3, 1, 2)
        plt.plot(df.index, df_freq[col], label=col, color='blue')
        plt.plot(df.index, df_freq[col + '_freq_weighted'], label = col + ' weighted frequency', color='yellow')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title('Weighted Frequency ' + col)

        plt.subplot(3, 1, 3)
        plt.plot(df.index, df_freq[col], label=col, color='blue')
        plt.plot(df.index, df_freq[col + '_pse'], label = col + ' pse', color='green')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title('PSE ' + col)

        plt.savefig(RESULT_PATH / RESULT_FNAME_PNG)
        plt.savefig(RESULT_PATH / RESULT_FNAME_PDF)

        plt.legend()
        plt.show()
        
# Specify the values needed for the abstraction, window size, the fuction to use and the columns to aggregate
windowSize = 120
cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'lin_acc_x','lin_acc_y','lin_acc_z','loc_speed', 'loc_horizontal_accuracy', 'loc_vertical_accuracy' , 'mang_field_x', 'mang_field_y' , 'mang_field_z', 'pca_1', 'pca_2', 'pca_3']
# cols = ['acc_x', 'gyr_x', 'lin_acc_x', 'loc_speed', 'mang_field_x']

df = pd.read_csv("Python3Code\phyphox-outputs\lowpass_extended_testing_results_pca.csv", skipinitialspace=True)

############################################# TEMPORAL ABSTRACTION ##########################################################

numAbs = NumericalAbstraction()

# Function used to aggregate the data 
functions = ['mean', 'max', 'min', 'median', 'std']
# functions = ['mean', 'max']

for func in functions:
    # Perform abstraction to get abstracted data
    df_abs = numAbs.abstract_numerical(df, cols, windowSize, func)
    # df_abs = windowOverlap(df_abs, windowSize)

    # Plot values
    # plotNumAbstraction(df_abs, cols, windowSize, func)

# print(df_abs.columns)

# file = Path('./Python3Code/temporal-feature-data/temporal_features_dataset_ws120.csv')
# file.parent.mkdir(parents=True, exist_ok=True)  
# df_abs.to_csv(file)

############################################# Classify Feature ###############################################################

# cols = ['loc_speed']
# col = 'loc_speed'

# def checkLow(x, low):
#     if x <= low:
#         return 1
#     else: 
#         return 0
    
# def checkMedium(x, low, medium):
#     if x > low and x <= medium:
#         return 1
#     else: 
#         return 0
    
# def checkHigh(x, medium, high):
#     if x > medium and x <= high:
#         return 1
#     else: 
#         return 0


# def labelFeatures(df, cols):

#     #Get all maximum/minimum values from dataframe
#     maximums = df_abs.max(axis='index')
#     minimums = df_abs.min(axis='index')

#     for col in cols:
#         # Get the maximum and minimum value of the current column
#         max = maximums[col]
#         min = minimums[col]

#         #Get the difference between the maximum and the minimum value for the column
#         range = abs(max - min)
#         unit = range/3

#         # Create boundaries for the low, medium and high values
#         low = min + unit
#         medium = min + 2*unit
#         high = min + 3*unit

#         # df['label_' + col] = df[col].apply(lambda x: labelValue(x, low, medium, high))

#         df['high_' + col] = df[col].apply(lambda x: checkLow(x, low))
#         df['medium_' + col] = df[col].apply(lambda x: checkMedium(x,low, medium))
#         df['low_' + col] = df[col].apply(lambda x: checkHigh(x, medium, high))

#     return df

# df = labelFeatures(df, cols)

# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df_abs[col], label=col, color='blue')
# plt.plot(df.index, df_abs['label_' + col], label=col, color='red')

# plt.xlabel('Index')
# plt.ylabel(col)
# plt.title('Differences between ' + col + ' and its Rolling '  + func)
# plt.legend()
# plt.show()



############################################# FREQUENCY ABSTRACTION ##########################################################

# Sample frequency (Hz)
fs = 160

df = df_abs

FreqAbs = FourierTransformation()

df_freq = FreqAbs.abstract_frequency(copy.deepcopy(df), cols, windowSize, fs)

# plotFreqAbstraction(df_freq, cols)

df_freq = windowOverlap(df_freq, windowSize)

file = Path('./Python3Code/frequency-feature-data/features_dataset_testing_ws120_fs160_overlap0.9.csv')
file.parent.mkdir(parents=True, exist_ok=True)  
df_freq.to_csv(file)






