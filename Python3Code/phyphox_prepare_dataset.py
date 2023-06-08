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
import pandas as pd

# Chapter 2: Initial exploration of the dataset.
# _USER_FOLDER = "marie-dataset"
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


# pprint(timestamped_datasets)
# timestamped_datasets ={
#     "walking 2023.csv:Accelerator.csv": dataset
#     "running 2023.csv:Accelerator.csv": dataset
#     "cycling 2023.csv:Accelerator.csv": dataset,
#     "walking 2023.csv:Gyposcope.csv": dataset,
# }

####################################################################
#Data Aggregation

# Define the activities and measurements
activities = ['walking', 'running', 'cycling', 'sitting', 'hammocking']
measurements = ['Accelerometer', 'Gyroscope', 'Light', 'Linear Acceleration', 'Location', 'Magnetometer']

# Iterate over each measurement
for measurement in measurements:

    # Create a dictionary to store the data for each measurement
    measurement_data = {}
    combined_measurement = pd.DataFrame()

    print('\n') 

    # Iterate over each activity
    for activity in activities:
        # Create an empty DataFrame to store the combined data for the activity
        combined_data = pd.DataFrame()

        print(measurement, ' ', activity)

        # Iterate over each file in the dictionary
        for file in timestamped_datasets:
            print(file)
            
            # if measurement in file and activity in file:
            #     # Read the Excel file into a DataFrame
            #     file_path = os.path.join(directory, file)
            #     df = pd.read_excel(file_path)

            # Check if the file contains the current measurement and activity
            if measurement in file and activity in file:
                print('True')
                #Get the data from the dictionary 
                data = timestamped_datasets[file]
                data.insert(len(data.columns), 'label', len(data.index)*[activity], True)
                # print(data)
                
                combined_data = combined_data.append(data)

        # print("combined data\n")
        # print(combined_data)
        combined_measurement = combined_measurement.append(combined_data)
    
    print("combined measurement\n")
    print(combined_measurement)
    # measurement_data[measurement] = combined_measurement
    # ['Accelerometer', 'Gyroscope', 'Light', 'Linear Acceleration', 'Location', 'Magnetometer']
    if (measurement == 'Accelerometer'):
        combined_measurement = combined_measurement.rename(columns={'Acceleration x (m/s^2)': 'x', 'Acceleration y (m/s^2)': 'y', 'Acceleration z (m/s^2)': 'z'})
    elif (measurement == 'Gyroscope'):
        combined_measurement.rename(columns={'Gyroscope x (rad/s)': 'x', 'Gyroscope y (rad/s)': 'y', 'Gyroscope z (rad/s)': 'z'}, inplace=True)
    elif (measurement == 'Light'):
        combined_measurement.rename(columns={'Illuminance (lx)': 'lux'}, inplace=True)
    elif (measurement == 'Linear Acceleration'):
        combined_measurement.rename(columns={'Linear Acceleration x (m/s^2)': 'x', 'Linear Acceleration y (m/s^2)': 'y', 'Linear Acceleration z (m/s^2)': 'z'}, inplace=True)
    elif (measurement == 'Location'):
        combined_measurement.rename(columns={'Latitude (Â°)': 'latitude', 'Longitude (Â°)': 'longitude', 'Velocity (m/s)': 'speed', 'Height (m)': 'height', 'Direction (Â°)': 'direction', 'Horizontal Accuracy (m)': 'horizontal_accuracy', 'Vertical Accuracy (m)': 'vertical_accuracy'}, inplace=True)
    elif (measurement == 'Magnetometer'):
        combined_measurement.rename(columns={'Magnetic field x (ÂµT)': 'x', 'Magnetic field y (ÂµT)': 'y', 'Magnetic field z (ÂµT)': 'z'}, inplace=True)
    else:
        pass

    print("combined measurement\n")
    print(combined_measurement)


    combined_measurement = combined_measurement.sort_values(by='timestamps')

    RESULT_FNAME = f'{measurement}.cvs'
    combined_measurement.to_csv(RESULT_PATH / RESULT_FNAME)

print('Done!')
