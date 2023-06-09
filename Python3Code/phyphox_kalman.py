from Chapter3.KalmanFilters import KalmanFilters
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys
import pandas as pd

#Kalman Filter object
kf = KalmanFilters()

#Get the data 
dataset = pd.read_csv("./phyphox-outputs/chapter2_result_250.csv", skipinitialspace=True)
kfDataset = pd.DataFrame()
# print(dataset)
# Plot the data
# DataViz = VisualizeDataset(__file__)

measurements = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'light_lux', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'loc_Latitude (°)', 'loc_Longitude (°)', 'loc_height', 'loc_speed',
                'loc_Direction (°)', 'loc_horizontal_accuracy', 'loc_vertical_accuracy', 'mang_Magnetic field x (µT)', 'mang_Magnetic field y (µT)', 'mang_Magnetic field z (µT)']

for measurement in measurements:
    # perform the Kalman Filter on the dataset
    kfDataset = kf.apply_kalman_filter(dataset, measurement)
    #Drop the initial measurement column
    kfDataset = kfDataset.drop([measurement], axis = 1)
    print(kfDataset)

RESULT_PATH = Path('./phyphox-outputs/')
RESULT_FNAME = f'kalman_filter_dataset.cvs'
kfDataset.to_csv(RESULT_PATH / RESULT_FNAME)











# if (measurement == 'Accelerometer'):
#     # combined_measurement = combined_measurement.rename(columns={'Acceleration x (m/s^2)': 'x', 'Acceleration y (m/s^2)': 'y', 'Acceleration z (m/s^2)': 'z'})
#     acc_x = kf.apply_kalman_filter(dataset, 'acc_x')
#     acc_y = kf.apply_kalman_filter(dataset, 'acc_y')
#     acc_z = kf.apply_kalman_filter(dataset, 'acc_z')

#     #INSERT DATA COLUM HERE 
#     # data.insert(len(data.columns), 'label', len(data.index)*[activity], True)

# elif (measurement == 'Gyroscope'):
#     # combined_measurement.rename(columns={'Gyroscope x (rad/s)': 'x', 'Gyroscope y (rad/s)': 'y', 'Gyroscope z (rad/s)': 'z'}, inplace=True)
#     gyro_x = kf.apply_kalman_filter(dataset, 'x')
#     gyro_y = kf.apply_kalman_filter(dataset, 'y')
#     gyro_z = kf.apply_kalman_filter(dataset, 'z')

# elif (measurement == 'Light'):
#     # combined_measurement.rename(columns={'Illuminance (lx)': 'lux'}, inplace=True)
#     light_lux = kf.apply_kalman_filter(dataset, 'lux')


# elif (measurement == 'Linear Acceleration'):
#     # combined_measurement.rename(columns={'Linear Acceleration x (m/s^2)': 'x', 'Linear Acceleration y (m/s^2)': 'y', 'Linear Acceleration z (m/s^2)': 'z'}, inplace=True)
#     linAcc_x = kf.apply_kalman_filter(dataset, 'x')
#     linAcc_y = kf.apply_kalman_filter(dataset, 'y')
#     linAcc_z = kf.apply_kalman_filter(dataset, 'z')

# elif (measurement == 'Location'):
#     # combined_measurement.rename(columns={'Latitude (Â°)': 'latitude', 'Longitude (Â°)': 'longitude', 'Velocity (m/s)': 'speed', 'Height (m)': 'height', 'Direction (Â°)': 'direction', 'Horizontal Accuracy (m)': 'horizontal_accuracy', 'Vertical Accuracy (m)': 'vertical_accuracy'}, inplace=True)
#     loc_lat = kf.apply_kalman_filter(dataset, 'latitude')
#     loc_long = kf.apply_kalman_filter(dataset, 'logitude')
#     loc_speed = kf.apply_kalman_filter(dataset, 'speed')
#     loc_h = kf.apply_kalman_filter(dataset, 'height')
#     loc_dir = kf.apply_kalman_filter(dataset, 'direction')
#     loc_hor = kf.apply_kalman_filter(dataset, 'horizontal_accuracy')
#     loc_ver = kf.apply_kalman_filter(dataset, 'vertical_accuracy')

# elif (measurement == 'Magnetometer'):
#     # combined_measurement.rename(columns={'Magnetic field x (ÂµT)': 'x', 'Magnetic field y (ÂµT)': 'y', 'Magnetic field z (ÂµT)': 'z'}, inplace=True)
#     mag_x = kf.apply_kalman_filter(dataset, 'x')
#     mag_y = kf.apply_kalman_filter(dataset, 'y')
#     mag_z = kf.apply_kalman_filter(dataset, 'z')









