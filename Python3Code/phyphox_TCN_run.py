
import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

# import keras
# from keras import Sequential
# from keras.layers import Dense
# from keras.models import Model
# from tcn import TCN
# from numpy import reshape
# from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
# from tensorflow.keras.layers import Reshape

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from util import util
from util.VisualizeDataset import VisualizeDataset


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tcn import TCN
from sklearn.preprocessing import LabelEncoder

# model = keras.models.load_model('./model/tcm.keras')
# model = tf.keras.saving.load_model('./model/tcm.keras')

# m = tf.keras.models.load_model('./model/tcm.keras')

model= tf.keras.models.load_model('./model/tcm.keras' , custom_objects={'TCN': TCN})

train_loss, train_acc = model.evaluate(X_train, y_train,
                                       batch_size=M_TRAIN, verbose=0)

train_loss, train_acc, _ = model.evaluate(X_train, y_train, batch_size=M_TRAIN, verbose=0)

test_loss, test_acc, _ = model.evaluate(X_test[:M_TEST], y_test[:M_TEST],
                                     batch_size=M_TEST, verbose=0)
print('-'*65)
print(f'train accuracy = {round(train_acc * 100, 4)}%')
print(f'test accuracy = {round(test_acc * 100, 4)}%')
print(f'test error = {round((1 - test_acc) * M_TEST)} out of {M_TEST} examples')