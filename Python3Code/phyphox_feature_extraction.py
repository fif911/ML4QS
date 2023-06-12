#Extract features from data

import pandas as pd
import copy
import matplotlib.pyplot as plt
import re
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction

# Sample frequency (Hz)
# fs = 10

df = pd.read_csv("Python3Code\phyphox-outputs\chapter2_result_250_labeled_aligned_min_extended_onehot_outlier.csv", skipinitialspace=True)
print(df.columns)

numAbs = NumericalAbstraction()

meanDT = numAbs.abstract_numerical(df, ['acc_x'], 2, 'mean')

print(meanDT)














# FreqAbs = FourierTransformation()
# data_table = FreqAbs.abstract_frequency(copy.deepcopy(df), ['acc_x', 'acc_y', 'acc_z'], 160, fs)
# print(data_table.columns)





# # Get the frequencies from the columns....
# frequencies = []
# values = []

# # Extracting the frequencies from columns obtained 
# for col in data_table.columns:
#     val = re.findall(r'freq_\d+\.\d+_Hz', col)
#     print(val)
#     if len(val) > 0:
#         frequency = float((val[0])[5:len(val)-4])
#         frequencies.append(frequency)
#         values.append(data_table.loc[data_table.index, col])

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# plt.xlim([0, 5])
# ax1.plot(frequencies, values, 'b+')
# ax1.set_xlabel('Frequency (Hz)')
# ax1.set_ylabel('$a$')
# plt.show()


