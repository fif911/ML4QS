
import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from keras.models import Model
from tcn import TCN
from numpy import reshape
from sklearn.preprocessing import LabelEncoder
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

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('Python3Code/frequency-feature-data/')

# name of dataset for training data
DATASET_FNAME = 'features_dataset_ws120_fs160_overlap0.9.csv'

# name of dataset for testing data
DATASET_TEST_FNAME = 'features_dataset_testing_ws120_fs160_overlap0.9.csv'

RESULT_FNAME = 'chapter7_classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/crowdsignals_ch7_classification/')

dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
dataset.index = pd.to_datetime(dataset.index)

test_dataset = pd.read_csv(DATA_PATH / DATASET_TEST_FNAME, index_col=0)
test_dataset.index = pd.to_datetime(test_dataset.index)

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.
dataset['class'] = dataset['label']
del dataset['label']
# dataset.index = dataset['time']
dataset = dataset.dropna()

feature_names = ['lin_acc_x_temp_std_ws_120', 'lin_acc_y_temp_std_ws_120', 'lin_acc_z_temp_std_ws_120',
                     "gyr_x_temp_std_ws_120",
                     "lin_acc_x_max_freq", 'loc_speed']
features = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if (x in feature_names)]
class_label_indices = [dataset.columns.get_loc(x) for x in dataset.columns if ('class' in x)]

train_X = dataset.iloc[:, features]
train_y = dataset.iloc[:, class_label_indices]

test_dataset['class'] = test_dataset['label']
del test_dataset['label']
# test_dataset.index = test_dataset['time']
test_dataset = test_dataset.dropna()
features = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if (x in feature_names)]
class_label_indices = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if ('class' in x)]

test_X = test_dataset.iloc[:, features]
test_y = test_dataset.iloc[:, class_label_indices]


X_train, y_train = [], []
T = 45
for i in range(train_y.shape[0] - (T-1)):
    X_train.append(train_X.iloc[i:i+T].values)
    y_train.append(train_y.iloc[i + (T-1)])
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)
print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

X_test, y_test = [], []
for i in range(test_y.shape[0] - (T-1)):
    X_test.append(test_X.iloc[i:i+T].values)
    y_test.append(test_y.iloc[i + (T-1)])

y_test = label_encoder.fit_transform(y_test)
X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1,1)
print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')

# classes = ['label_walking', 'label_running', 'label_hammocking', 'label_sitting', 'label_cycling']


# Let's make a list of CONSTANTS for modelling:
LAYERS = [8, 8, 8, 1]                # number of units in hidden and output layers
M_TRAIN = X_train.shape[0]           # number of training examples (2D)
M_TEST = X_test.shape[0]             # number of test examples (2D),full=X_test.shape[0]
N = X_train.shape[2]                 # number of features
BATCH = M_TRAIN                      # batch size
EPOCH = 50                       # number of epochs
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate
print(f'layers={LAYERS}, train_examples={M_TRAIN}, test_examples={M_TEST}')
print(f'batch = {BATCH}, timesteps = {T}, features = {N}, epochs = {EPOCH}')
print(f'lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}')

# ... Your code for loading and preprocessing the data ...

num_classes = 5
# y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_train = np.repeat(y_train[:, np.newaxis, :], T, axis=1)  # Repeat labels along the time axis

y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
y_test = np.repeat(y_test[:, np.newaxis, :], T, axis=1)  # Repeat labels along the time axis


# Reshape the input data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# y_train = keras.utils.to_categorical(y_train, num_classes=5)
# y_test = keras.utils.to_categorical(y_test, num_classes=5)

# # Define the TCN model
model = Sequential([
    TCN(input_shape=(T, N), nb_filters=256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32, 64, 128]),
    Dense(num_classes, activation='softmax')
])


# # Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# # Train the model
# History = model.fit(X_train, y_train, epochs=EPOCH)
History = model.fit(X_train, y_train, epochs=EPOCH, validation_data=(X_test, y_test))

print(History.history)

# model.save('./model/tcm.keras')

# model= tf.keras.models.load_model('./model/tcm.keras' , custom_objects={'TCN': TCN})

train_loss, train_acc, _ = model.evaluate(X_train, y_train, batch_size=M_TRAIN, verbose=0)

test_loss, test_acc, _ = model.evaluate(X_test[:M_TEST], y_test[:M_TEST],
                                     batch_size=M_TEST, verbose=0)
print('-'*65)
print(f'train accuracy = {round(train_acc * 100, 4)}%')
print(f'test accuracy = {round(test_acc * 100, 4)}%')
print(f'test error = {round((1 - test_acc) * M_TEST)} out of {M_TEST} examples')


# # Plot the loss and accuracy curves over epochs:
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
axs[0].plot(History.history['loss'], color='b', label='Training loss')
axs[0].plot(History.history['val_loss'], color='r', label='Validation loss')
axs[0].set_title("Loss curves")
axs[0].legend(loc='best', shadow=True)
axs[1].plot(History.history['accuracy'], color='b', label='Training accuracy')
axs[1].plot(History.history['val_accuracy'], color='r', label='Validation accuracy')
axs[1].set_title("Accuracy curves")
axs[1].legend(loc='best', shadow=True)

plt.show()





# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

# # Make predictions
# predictions = model.predict(X_test)
# predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))


# loss, accuracy, categorical_accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
# print("Test Categorical Accuracy:", categorical_accuracy)