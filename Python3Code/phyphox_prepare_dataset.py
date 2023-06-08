##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# Chapter 2: Initial exploration of the dataset.
_USER_FOLDER = "alex-test-dataset"
# User folder have multiple experiments
# Each experiment have multiple files (Accelerometer.csv, Gyroscope.csv, etc)

_EXPERIMENT_NAME = 'Walking 2030'

USER_PATH = Path(f'./datasets/phyphox/{_USER_FOLDER}')

RESULT_PATH = Path('./phyphox-outputs/')
RESULT_FNAME = f'{_USER_FOLDER}_AGGREGATED_BY_TYPE.csv' # TODO: Add type


# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [RESULT_PATH]]

print('Please wait, this will take a while to run!')

timestamped_datasets = {}

print("Preparing datasets...")

experiment_names = [f for f in os.listdir(USER_PATH) if os.path.isdir(USER_PATH / f)]

for experiment_name in experiment_names:

    for file_name in os.listdir(USER_PATH / experiment_name):
        # get timestamp offset from meta/time.csv
        timestamp_offset = CreateDataset.get_timestamp_offset(USER_PATH / experiment_name)

        # Ignore meta files
        if file_name not in ["meta","device.csv","time.csv"]:
            print(f"Preparing {file_name}")
            dataset = CreateDataset.prepare_dataset(USER_PATH / experiment_name / file_name, timestamp_offset)
            timestamped_datasets[f"{experiment_name}:{file_name}"] = dataset
        else:
            print(f"Skipping {file_name}")

exit(0)
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    dataset.add_numerical_dataset('Accelerometer.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'acc_phone_')
    # dataset.add_numerical_dataset('accelerometer_smartwatch.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'acc_watch_')
    #
    # # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
    # # and aggregate the values per timestep by averaging the values
    # dataset.add_numerical_dataset('gyroscope_phone.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyr_phone_')
    # dataset.add_numerical_dataset('gyroscope_smartwatch.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyr_watch_')
    #
    # # We add the heart rate (continuous numerical measurements) and aggregate by averaging again
    # dataset.add_numerical_dataset('heart_rate_smartwatch.csv', 'timestamps', ['rate'], 'avg', 'hr_watch_')
    #
    # # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # # occurs within an interval).
    # dataset.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')
    #
    # # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging
    # dataset.add_numerical_dataset('light_phone.csv', 'timestamps', ['lux'], 'avg', 'light_phone_')
    #
    # # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
    # # and aggregate the values per timestep by averaging the values
    # dataset.add_numerical_dataset('magnetometer_phone.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'mag_phone_')
    # dataset.add_numerical_dataset('magnetometer_smartwatch.csv', 'timestamps', ['x', 'y', 'z'], 'avg', 'mag_watch_')
    #
    # # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
    # dataset.add_numerical_dataset('pressure_phone.csv', 'timestamps', ['pressure'], 'avg', 'press_phone_')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y',
                                           'acc_watch_z'])

    # Plot all data
    DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
                         ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

# Make a table like the one shown in the book, comparing the two datasets produced.
util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through

print('The code has run through successfully!')
