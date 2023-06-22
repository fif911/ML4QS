from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
from hyperopt import Trials, STATUS_OK, tpe
from keras.regularizers import l2

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# window for data agregation

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    # Read the result from the previous chapter, and make sure the index is of the type datetime.
    DATA_PATH = Path('./frequency-feature-data/')

    # name of dataset for training data
    DATASET_FNAME = 'features_dataset_ws120_fs160_overlap0.9.csv'

    # name of dataset for testing data
    DATASET_TEST_FNAME = 'features_dataset_testing_ws120_fs160_overlap0.9.csv'
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
    for i in range(train_y.shape[0] - (45-1)):
        X_train.append(train_X.iloc[i:i+45].values)
        y_train.append(train_y.iloc[i + (45-1)])
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)
    print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

    X_test, y_test = [], []
    for i in range(test_y.shape[0] - (45-1)):
        X_test.append(test_X.iloc[i:i+45].values)
        y_test.append(test_y.iloc[i + (45-1)])

    y_test = label_encoder.fit_transform(y_test)
    X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1,1)
    print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')
    return X_train, y_train, X_test, y_test


def create_model(X_train, y_train, X_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    model = Sequential()
    model.add(LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=False, input_shape=(45, X_train.shape[2])))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(units=X_train.shape[2]))
    model.add(Activation({{choice(['relu', 'tanh', 'sigmoid', 'softmax'])}}))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate={{choice( [0.1, 0.01, 0.001, 0.0001, 0.00001])}}), metrics=['accuracy'])

    result = model.fit(X_train, y_train, 
                       batch_size={{choice([16, 32, 64, 128, 256])}}, 
                       epochs={{choice([10, 50, 100])}}, 
                       validation_data=(X_test, y_test), 
                       verbose=2)

    print(result.history['accuracy'])
    print(result.history)

    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    # best_run, best_model = optim.minimize(model=create_model,
    #                                       data=data,
    #                                       algo=tpe.suggest, # try with rand.suggest
    #                                       max_evals=5,
    #                                       trials=Trials())
    X_train, y_train, X_test, y_test = data()

    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(X_test, y_test))
    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)
    # best_model.save('./model/lstm.keras')

    loaded_model = tf.keras.saving.load_model('./model/lstm.keras')
    M_TEST = X_test.shape[0]

    History = loaded_model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_data=(X_test[:M_TEST], y_test[:M_TEST]))

 

    train_loss, train_acc = loaded_model.evaluate(X_train, y_train,
                                       batch_size=128, verbose=0)
    test_loss, test_acc = loaded_model.evaluate(X_test[:M_TEST], y_test[:M_TEST],
                                        batch_size=M_TEST, verbose=0)
    print('-'*65)
    print(f'train accuracy = {round(train_acc * 100, 4)}%')
    print(f'test accuracy = {round(test_acc * 100, 4)}%')
    print(f'test error = {round((1 - test_acc) * M_TEST)} out of {M_TEST} examples')

    # Plot the loss and accuracy curves over epochs:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    axs[0].plot(History.history['loss'], color='b', label='Training loss')
    axs[0].plot(History.history['val_loss'], color='r', label='Validation loss')
    axs[0].set_title("Loss curves")
    axs[0].legend(loc='best', shadow=True)
    axs[1].plot(History.history['accuracy'], color='b', label='Training accuracy')
    axs[1].plot(History.history['val_accuracy'], color='r', label='Validation accuracy')
    axs[1].set_title("Accuracy curves")
    axs[1].legend(loc='best', shadow=True)
    # plt.savefig("./figures/phyphox_hyperas/fig1.png")
    # plt.savefig("./figures/phyphox_hyperas/fig1.pdf")
    plt.show()