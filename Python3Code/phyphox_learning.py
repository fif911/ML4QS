##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

start = time.time()

from sklearn.model_selection import train_test_split

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from util import util
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./frequency-feature-data/')

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
dataset['class'] = dataset['label']
del dataset['label']
# dataset.index = dataset['time']
dataset = dataset.dropna()

features = [dataset.columns.get_loc(x) for x in dataset.columns if
            ('label' not in x) and ('class' not in x) and ('time' not in x)]
class_label_indices = [dataset.columns.get_loc(x) for x in dataset.columns if ('class' in x)]

train_X = dataset.iloc[:, features]
train_y = dataset.iloc[:, class_label_indices]

test_dataset['class'] = test_dataset['label']
del test_dataset['label']
# test_dataset.index = test_dataset['time']
test_dataset = test_dataset.dropna()
features = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if
            ('label' not in x) and ('class' not in x) and ('time' not in x)]
class_label_indices = [test_dataset.columns.get_loc(x) for x in test_dataset.columns if ('class' in x)]

test_X = test_dataset.iloc[:, features]
test_y = test_dataset.iloc[:, class_label_indices]

print('Training set length is: ', len(train_X.index))
print('Test set length is: ', len(test_X.index))

# Select subsets of the features that we will consider:

basic_features = ['acc_x', 'acc_y', 'acc_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'loc_speed', 'gyr_x', 'gyr_y',
                  'gyr_z', 'loc_horizontal_accuracy', 'loc_vertical_accuracy', 'mang_field_x', 'mang_field_y',
                  'mang_field_z']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_features = [name for name in dataset.columns if '_temp_' in name]
freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
print('#basic features: ', len(basic_features))
print('#PCA features: ', len(pca_features))
print('#time features: ', len(time_features))
print('#frequency features: ', len(freq_features))
features_after_chapter_3 = list(set().union(basic_features, pca_features))
features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features))

# # First, let us consider the performance over a selection of features:

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 50

#  ---------------------------- FORWARD SELECTION ---------------------------------

# fs = FeatureSelectionClassification()
# features, ordered_features, ordered_scores = fs.forward_selection(N_FORWARD_SELECTION,
#                                                                   train_X[features_after_chapter_5],
#                                                                   test_X[features_after_chapter_5],
#                                                                   train_y,
#                                                                   test_y,
#                                                                   gridsearch=False)
# _dataset = pd.DataFrame()
# _dataset['features'] = np.array(ordered_features)
# _dataset['scores'] = np.array(ordered_scores)

# print(_dataset.sort_values('scores'))

# _dataset.to_csv("phyphox-outputs/feature_selection.csv")

# DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION + 1)], y=[ordered_scores],
#                 xlabel='number of features', ylabel='accuracy')

#  ---------------------------- PEARSON SELECTION ---------------------------------

# Tried to run pearson selection but not conclusive: all correlations have coefficient 1 
# something must have gone wrong with the class computation for example

# fsr = FeatureSelectionRegression()
# formatted_y_train = copy.deepcopy(train_y)
# dict = {'cycling': 1.0, 'hammocking': 2.0, 'sitting': 3.0, 'running': 4.0, 'walking': 5.0}
# formatted_y_train['class'] = formatted_y_train['class'].apply(lambda x: dict[x])
# print(formatted_y_train)
# res_list, ordered_res = fsr.pearson_selection(N_FORWARD_SELECTION, train_X, formatted_y_train)
# print(res_list)
# print(ordered_res)

# res_list.to_csv("phyphox-outputs/feature_selection_pearson.csv")

# DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION + 1)], y=[ordered_res],
#                 xlabel='number of features', ylabel='accuracy')

#  -----------------------------------------------------

reduced_features = ["lin_acc_x", "loc_speed"]
selected_features = ['lin_acc_x_temp_std_ws_120', 'lin_acc_y_temp_std_ws_120', 'lin_acc_z_temp_std_ws_120',
                     "gyr_x_temp_std_ws_120",
                     "lin_acc_x_max_freq", 'loc_speed']

"""
acc_x_temp_std_ws_120,
acc_y_temp_std_ws_120,
acc_z_temp_std_ws_120
,gyr_x_temp_std_ws_120
,gyr_y_temp_std_ws_120,
gyr_z_temp_std_ws_120,
lin_acc_x_temp_std_ws_120,
lin_acc_y_temp_std_ws_120,
lin_acc_z_temp_std_ws_120,
loc_speed_temp_std_ws_120,
loc_horizontal_accuracy_temp_std_ws_120,
loc_vertical_accuracy_temp_std_ws_120,
mang_field_x_temp_std_ws_120,
mang_field_y_temp_std_ws_120,
mang_field_z_temp_std_ws_120,
pca_1_temp_std_ws_120,
pca_2_temp_std_ws_120,
pca_3_temp_std_ws_120,acc_x_max_freq
"""

# Drop PCA and accuracy columns
all_reduced_features = [col for col in features_after_chapter_5 if 'pca' not in col]
all_reduced_features = [col for col in all_reduced_features if 'accuracy' not in col]
print(
    f"Reduced number of features after chapter 5 (some dropped): {len(all_reduced_features)}/{len(features_after_chapter_5)}")

features = [basic_features, reduced_features, selected_features, all_reduced_features, features_after_chapter_5]
feature_names = ['basic_features', "basic_reduced_features", 'selected features', "all_reduced_features",
                 "all_features"]

learner = ClassificationAlgorithms()
eval = ClassificationEvaluation()

# ---------------------------- TUNING PARAMETERS ----------------------------------------
# by setting gridsearch to true, the optimal parameters are returned (i changed the original code to return them)

# params = learner.decision_tree(
#         train_X[selected_features], train_y, test_X[selected_features],
#         gridsearch=True, print_model_details=True)

# params = learner.feedforward_neural_network(
#             train_X[selected_features], train_y, test_X[selected_features], gridsearch=True
#         )

# params = learner.random_forest(
#             train_X[selected_features], train_y, test_X[selected_features], gridsearch=True
#         )

# params = learner.support_vector_machine_with_kernel(
#             train_X[selected_features], train_y, test_X[selected_features], gridsearch=True
#         )

# params = learner.support_vector_machine_without_kernel(
#             train_X[selected_features], train_y, test_X[selected_features], gridsearch=True
#         )

# params = learner.k_nearest_neighbor(
#         train_X[selected_features], train_y, test_X[selected_features], gridsearch=True
#     )

# print(params)

SKIP_COMPARING = False
if not SKIP_COMPARING:
    #  ---------------------------- COMPARING LEARNING ALGORITHMS ----------------------------
    N_KCV_REPEATS = 5
    scores_over_all_algs = []

    for j, feature_set in enumerate(features):
        performance_tr_dt = 0
        performance_te_dt = 0
        performance_tr_nn = 0
        performance_te_nn = 0
        performance_tr_rf = 0
        performance_te_rf = 0
        performance_tr_sv = 0
        performance_te_sv = 0
        performance_tr_so = 0
        performance_te_so = 0
        # performance_tr_kn = 0
        # performance_te_kn = 0
        performance_tr_nb = 0
        performance_te_nb = 0

        for i in range(N_KCV_REPEATS):
            print('perfoming neural network', i)
            # not enough features if we only take the selected (does not converge)
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
                train_X[feature_set], train_y, test_X[feature_set],
                gridsearch=False,
                hidden_layer_sizes=(100,),
                activation='logistic',
                learning_rate='adaptive',
                max_iter=2000,
                alpha=0.0001
            )
            performance_tr_nn += eval.accuracy(train_y, class_train_y)
            performance_te_nn += eval.accuracy(test_y, class_test_y)

            print('perfoming random forest')
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                train_X[feature_set], train_y, test_X[feature_set],
                gridsearch=False,
                min_samples_leaf=2,
                n_estimators=100,
                criterion='gini'
            )
            performance_tr_rf += eval.accuracy(train_y, class_train_y)
            performance_te_rf += eval.accuracy(test_y, class_test_y)

            print('performing support vector machine with kernel')
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_with_kernel(
                train_X[feature_set], train_y, test_X[feature_set],
                gridsearch=False,
                kernel='rbf',
                gamma='scale',  # not sure about this one
                C=1.0
            )

            performance_tr_sv += eval.accuracy(train_y, class_train_y)
            performance_te_sv += eval.accuracy(test_y, class_test_y)

            print('performing support vector machine without kernel')
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.support_vector_machine_without_kernel(
                train_X[feature_set], train_y, test_X[feature_set],  # does not converge with selected features
                gridsearch=False,
                max_iter=1000,
                tol=0.0001,
                C=1.0
            )
            performance_tr_so += eval.accuracy(train_y, class_train_y)
            performance_te_so += eval.accuracy(test_y, class_test_y)

        print('perfoming decision tree')
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            train_X[feature_set], train_y, test_X[feature_set],
            gridsearch=False,
            criterion='gini',  # best parameter according to grid search
            min_samples_leaf=2
        )
        performance_tr_dt += eval.accuracy(train_y, class_train_y)
        performance_te_dt += eval.accuracy(test_y, class_test_y)

        # print('performing k nearest neighbors')
        # class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.k_nearest_neighbor(
        #         train_X[feature_set], train_y, test_X[feature_set],
        #         gridsearch=True,
        #         # n_neighbors=5,
        #     )
        # performance_tr_kn += eval.accuracy(train_y, class_train_y)
        # performance_te_kn += eval.accuracy(test_y, class_test_y)

        print('performing naive bayes')
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.naive_bayes(
            train_X[feature_set], train_y, test_X[feature_set],
        )
        performance_tr_nb += eval.accuracy(train_y, class_train_y)
        performance_te_nb += eval.accuracy(test_y, class_test_y)

        scores_with_sd = util.print_table_row_performances(feature_names[j], len(train_X[feature_set].index),
                                                           len(test_X[feature_set].index), [
                                                               (performance_tr_nn / N_KCV_REPEATS,
                                                                performance_te_nn / N_KCV_REPEATS),
                                                               (performance_tr_rf / N_KCV_REPEATS,
                                                                performance_te_rf / N_KCV_REPEATS),
                                                               (performance_tr_sv / N_KCV_REPEATS,
                                                                performance_te_sv / N_KCV_REPEATS),
                                                               (performance_tr_so / N_KCV_REPEATS,
                                                                performance_te_so / N_KCV_REPEATS),
                                                               # (performance_tr_kn , performance_te_kn  ),
                                                               (performance_tr_dt, performance_te_dt),
                                                               (performance_tr_nb, performance_te_nb)])

        scores_over_all_algs.append(scores_with_sd)

    # DataViz.plot_performances_classification(['NN', 'RF', 'SVMK', 'SVM', 'KNN', 'DT', 'NB'], feature_names, scores_over_all_algs)
    DataViz.plot_performances_classification(['NN', 'RF', 'SVMK', 'SVM', 'DT', 'NB'], feature_names,
                                             scores_over_all_algs,
                                             lower_bound=0)


############### Study algorithms in detail
# We study NN for selected features and NB for basic_reduced_features

# NN for basic_reduced_features

learners = {}
# selected_features = ['lin_acc_x_temp_std_ws_120', "gyr_x_temp_std_ws_120", "lin_acc_x_max_freq", 'loc_speed']
#
# # 0.925
# learners["conf1"] = {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (5,), 'learning_rate': 'adaptive', 'max_iter': 1500}
# learners["conf2"] = {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100,10, 10), 'learning_rate': 'adaptive', 'max_iter': 3000}
learners["NN selected features conf1"] = {"hidden_layer_sizes": (100,),
                                          "activation": 'logistic',
                                          "learning_rate": 'adaptive',
                                          "max_iter": 3000,
                                          "alpha": 0.0001}
# learners["NN selected features conf2"] = {"hidden_layer_sizes": (100),
#                                           "activation": 'logistic',
#                                           "learning_rate": 'adaptive',
#                                           "max_iter": 4000,
#                                           "alpha": 0.0001}
#
for name, settings in learners.items():
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.feedforward_neural_network(
        train_X[selected_features], train_y, test_X[selected_features],
        # gridsearch=True,
        print_model_details=True,
        **settings
    )

    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)
    print(f"Accuracy for {name}: {eval.accuracy(test_y, class_test_y)}")

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False,
                                  title=f"Confusion matrix: {name}")



#################### Plot actual label and predicted label ####################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

pred_label = pd.DataFrame(class_test_y, columns=['pred_label'])
pred_label.index = test_y.index
dataset = pd.concat([test_y, pred_label], axis=1)

cl0 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])])
cl1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])])

labels = cl0.fit_transform(dataset).toarray()
dataset['label_cycling'] = labels[:, 0]
dataset['label_hammocking'] = labels[:, 1]
dataset['label_running'] = labels[:, 2]
dataset['label_sitting'] = labels[:, 3]
dataset['label_walking'] = labels[:, 4]

labels = cl1.fit_transform(dataset).toarray()
dataset['pred_cycling'] = labels[:, 0]
dataset['pred_hammocking'] = labels[:, 1]
dataset['pred_running'] = labels[:, 2]
dataset['pred_sitting'] = labels[:, 3]
dataset['pred_walking'] = labels[:, 4]

del dataset['class']
del dataset['pred_label']

DataViz.plot_dataset(dataset,
                     ['label_', 'pred_'],
                     ['like', 'like'],
                     ['points', 'points']
                     )
