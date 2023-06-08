##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################
from pprint import pprint

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# Chapter 2: Initial exploration of the dataset.
_USER_FOLDER = "marie-dataset"
# User folder have multiple experiments
# Each experiment have multiple files (Accelerometer.csv, Gyroscope.csv, etc)

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


pprint(timestamped_datasets)
# timestamped_datasets ={
#     "walking 2023.csv:Accelerator.csv": dataset
#     "running 2023.csv:Accelerator.csv": dataset
#     "cycling 2023.csv:Accelerator.csv": dataset,
#     "walking 2023.csv:Gyposcope.csv": dataset,
# }

####################################################################
#Data Aggregation

# print(timestamped_datasets)
#Dictionary keys - the files of the datasets
files = timestamped_datasets.keys()

print(files[0])
exit(0)

# Define the activities and measurements
activities = ['walking', 'running', 'cycling', 'sitting', 'hammocking']
measurements = ['Accelerometer', 'Gyroscope', 'Light', 'Linear Acceleration', 'Location', 'Magnetometer', 'Proximity']

# Iterate over each measurement
for measurment in measurements:

    # Create a dictionary to store the data for each activity
    activity_data = {}

    # Iterate over each activity
    for activity in activities:
        # Create an empty DataFrame to store the combined data for the activity
        # combined_data = pd.DataFrame()

        print(measurment, ' ', activity)
        # # Iterate over each Excel file
        # for file in excel_files:
        #     # Check if the file contains the current measurement and activity
        #     if measurement in file and activity in file:
        #         # Read the Excel file into a DataFrame
        #         file_path = os.path.join(directory, file)
        #         df = pd.read_excel(file_path)

        #         # Append the data to the combined DataFrame
        #         combined_data = combined_data.append(df)

        # Store the combined data for the activity in the dictionary
    #     activity_data[activity] = combined_data

    # # Create a new Excel file for each measurement
    # measurement_file = f'{measurement}_data.xlsx'
    # writer = pd.ExcelWriter(measurement_file, engine='xlsxwriter')

    # # Write each activity's data to a separate sheet in the Excel file
    # for activity, data in activity_data.items():
    #     data.to_excel(writer, sheet_name=activity, index=False)

    # # Save and close the Excel file
    # writer.save()
    # writer.close()

    # print(f'{measurement} data has been saved to {measurement_file}.')

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
