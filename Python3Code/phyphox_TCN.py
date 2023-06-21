
import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

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


# Data preprocessing

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('Python3Code/frequency-feature-data/')

# name of dataset for training data
# TODO add all features
DATASET_FNAME = 'features_dataset_ws120_fs160_overlap0.9.csv'

# name of dataset for testing data
# TODO add all features
DATASET_TEST_FNAME = 'features_dataset_testing_ws120_fs160_overlap0.9.csv'

RESULT_FNAME = 'chapter7_classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/crowdsignals_ch7_classification/')

dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
dataset.index = pd.to_datetime(dataset.index)

test_dataset = pd.read_csv(DATA_PATH / DATASET_TEST_FNAME, index_col=0)
test_dataset.index = pd.to_datetime(test_dataset.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# We create a single column with the categorical attribute representing our class. Furthermore, we use 70% of our data
# for training and the remaining 30% as an independent test set. We select the sets based on stratified sampling. We remove
# cases where we do not know the label.

##################### Training Set #################
dataset['class'] = dataset['label']
del dataset['label']

# dataset.index = dataset['time']
dataset = dataset.dropna()

features = [dataset.columns.get_loc(x) for x in dataset.columns if
            ('label' not in x) and ('class' not in x) and ('time' not in x)]

class_label_indices = [dataset.columns.get_loc(x) for x in dataset.columns if ('class' in x)]

train_X = dataset.iloc[:, features]
train_y = dataset.iloc[:, class_label_indices]

##################### Test Set #####################

test_dataset['class'] = test_dataset['label']
del test_dataset['label']

# test_dataset.index = test_dataset['time']
test_dataset = test_dataset.dropna()
features = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if
            ('label' not in x) and ('class' not in x) and ('time' not in x)]
class_label_indices = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if ('class' in x)]

test_X = test_dataset.iloc[:, features]
test_y = test_dataset.iloc[:, class_label_indices]

# print('Training set length is: ', len(train_X.index))
# print('Test set length is: ', len(test_X.index))

len_train = len(train_X.index)
len_test = len(test_X.index)


###########################################################################################################

# Select subsets of the features that we will consider:

# basic_features = ['acc_x', 'acc_y', 'acc_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'loc_speed', 'gyr_x', 'gyr_y',
#                   'gyr_z', 'loc_horizontal_accuracy', 'loc_vertical_accuracy', 'mang_field_x', 'mang_field_y',
#                   'mang_field_z']
# pca_features = ['pca_1', 'pca_2', 'pca_3']
# time_features = [name for name in dataset.columns if '_temp_' in name]
# freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]

# print('#basic features: ', len(basic_features))
# print('#PCA features: ', len(pca_features))
# print('#time features: ', len(time_features))
# print('#frequency features: ', len(freq_features))

# total_Features = len(basic_features) + len(pca_features) + len(time_features) + len(freq_features)


# features_after_chapter_3 = list(set().union(basic_features, pca_features))
# features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
# features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features))


# Obtaining total number of instances, features, and classes


###########################################################################################################


# labels = dataset[['label_walking', 'label_running', 'label_hammocking', 'label_sitting', 'label_cycling']]

# # # Convert labels to numerical values
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels.idxmax(axis=1))



# # Model training
# model = keras.models.Sequential([
#     TCN(input_shape=(num_instances, num_features), nb_filters=256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32]),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer="adam", loss="mae", metrics=keras.metrics.MeanAbsoluteError())



classes = ['label_walking', 'label_running', 'label_hammocking', 'label_sitting', 'label_cycling']

label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)



num_instances = len(train_X.index)
num_features = len(features)
num_classes = len(classes)



# train_X = train_X.reshape(-1, num_instances, num_features)
# train_y_encoded = train_y_encoded.reshape(-1,)
print(train_X.shape)
print(train_y.shape)

# Model training
model = keras.models.Sequential([
    # keras.layers.Flatten(input_shape=(num_features, num_instances)),
    # TCN(input_shape=(num_instances, num_features), nb_filters=256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32]),
    # reshape((num_instances, num_features)),
    # keras.layers.Flatten(),
    # keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_X, train_y_encoded, epochs=3)

# n_epochs = 1

# history = model.fit(train_X, train_y, 
#                         validation_data=(test_X, test_y), 
#                         epochs=n_epochs, 
#                         batch_size=num_instances)

# tcn_full_summary(m, expand_residual_blocks=False)

# x, y = get_x_y()
# m.fit(x, y, epochs=10, validation_split=0.2)






# # Step 5: Model evaluation
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

# history = model.fit(X_train, y_train, 
#                     validation_data=(X_valid, y_valid), 
#                     epochs=n_epochs, 
#                     batch_size=1024, 
#                     callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])

# model.save(f'Fold{fold+1} weights')
# test_preds.append(model.predict(test_data).squeeze().reshape(-1, 1).squeeze())







################################################
# # Step 6: Make predictions
# new_data = pd.read_csv('new_data.csv')  # Load new, unseen data
# new_features = new_data[['acc_x', 'acc_y', 'acc_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
#                          'gyr_x', 'gyr_y', 'gyr_z']]

# # Normalize new features
# new_features = scaler.transform(new_features)

# # Make predictions
# predictions = model.predict(new_features)
# predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
# print("Predicted Labels:", predicted_labels)


























