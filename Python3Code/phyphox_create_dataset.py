# Import the relevant classes.
import copy
import time
from multiprocessing import Process
from pathlib import Path

from Chapter2.CreateDataset import CreateDataset
from util import util
from util.VisualizeDataset import VisualizeDataset

# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/phyphox/prepared-marie')
RESULT_PATH = Path('./phyphox-outputs/')

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

print('Please wait, this will take a around 30 minutes to run!')
start_time = time.time()

datasets = []
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    # p_acc = Process(target=dataset.add_numerical_dataset,
    #                 args=('Accelerometer.cvs', 'timestamps', ['x', 'y', 'z'], 'avg', 'acc_',))
    # p_gyr = Process(target=dataset.add_numerical_dataset,
    #                 args=('Gyroscope.cvs', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyr_',))

    dataset.add_numerical_dataset('Accelerometer.cvs', 'timestamps', ['x', 'y', 'z', 'label'], 'avg', 'acc_')
    dataset.add_numerical_dataset('Gyroscope.cvs', 'timestamps', ['x', 'y', 'z'], 'avg', 'gyr_')

    dataset.add_numerical_dataset('Linear Acceleration.cvs', 'timestamps', ['x', 'y', 'z'], 'avg',
                                  'lin_acc_')
    dataset.add_numerical_dataset('Location.cvs', 'timestamps',
                                  ['Latitude (°)', 'Longitude (°)', 'height', 'speed', 'Direction (°)',
                                   'horizontal_accuracy', 'vertical_accuracy'], 'avg', 'loc_')
    dataset.add_numerical_dataset('Magnetometer.cvs', 'timestamps',
                                  ['Magnetic field x (µT)', 'Magnetic field y (µT)', 'Magnetic field z (µT)'], 'avg',
                                  'mang_')

    # # Get the resulting pandas data table
    dataset = dataset.data_table

    # # Plot the data
    # DataViz = VisualizeDataset(__file__)
    #
    # # Boxplot
    # DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y',
    #                                        'acc_watch_z'])
    #
    # # Plot all data
    # DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
    #                      ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
    #                      ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    # datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}_labeled.csv')
    print(f"Saved to csv as {RESULT_PATH}/chapter2_result_{milliseconds_per_instance}_labeled.csv")

# Make a table like the one shown in the book, comparing the two datasets produced.
# util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Lastly, print a statement to know the code went through
end_time = time.time() - start_time
print(f'The code took {end_time / 60} minutes to run')
print('The code has run through successfully!')
