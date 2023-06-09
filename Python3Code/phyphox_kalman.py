from pykalman import KalmanFilter

from Chapter3.KalmanFilters import KalmanFilters
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys
import pandas as pd

kf = KalmanFilters()

#Get the data 
data = []

#Go through each measurement, then through each table in the measurement


measurements = ['Accelerometer', 'Gyroscope', 'Light', 'Linear Acceleration', 'Location', 'Magnetometer']

measurement = 'Accelerometer'

if (measurement == 'Accelerometer'):
    # combined_measurement = combined_measurement.rename(columns={'Acceleration x (m/s^2)': 'x', 'Acceleration y (m/s^2)': 'y', 'Acceleration z (m/s^2)': 'z'})
    acc_x = kf.apply_kalman_filter(data, 'x')
    acc_y = kf.apply_kalman_filter(data, 'y')
    acc_z = kf.apply_kalman_filter(data, 'z')

    #INSERT DATA COLUM HERE 
    # data.insert(len(data.columns), 'label', len(data.index)*[activity], True)

elif (measurement == 'Gyroscope'):
    # combined_measurement.rename(columns={'Gyroscope x (rad/s)': 'x', 'Gyroscope y (rad/s)': 'y', 'Gyroscope z (rad/s)': 'z'}, inplace=True)
    gyro_x = kf.apply_kalman_filter(data, 'x')
    gyro_y = kf.apply_kalman_filter(data, 'y')
    gyro_z = kf.apply_kalman_filter(data, 'z')

elif (measurement == 'Light'):
    # combined_measurement.rename(columns={'Illuminance (lx)': 'lux'}, inplace=True)
    light_lux = kf.apply_kalman_filter(data, 'lux')


elif (measurement == 'Linear Acceleration'):
    # combined_measurement.rename(columns={'Linear Acceleration x (m/s^2)': 'x', 'Linear Acceleration y (m/s^2)': 'y', 'Linear Acceleration z (m/s^2)': 'z'}, inplace=True)
    linAcc_x = kf.apply_kalman_filter(data, 'x')
    linAcc_y = kf.apply_kalman_filter(data, 'y')
    linAcc_z = kf.apply_kalman_filter(data, 'z')

elif (measurement == 'Location'):
    # combined_measurement.rename(columns={'Latitude (Â°)': 'latitude', 'Longitude (Â°)': 'longitude', 'Velocity (m/s)': 'speed', 'Height (m)': 'height', 'Direction (Â°)': 'direction', 'Horizontal Accuracy (m)': 'horizontal_accuracy', 'Vertical Accuracy (m)': 'vertical_accuracy'}, inplace=True)
    loc_lat = kf.apply_kalman_filter(data, 'latitude')
    loc_long = kf.apply_kalman_filter(data, 'logitude')
    loc_speed = kf.apply_kalman_filter(data, 'speed')
    loc_h = kf.apply_kalman_filter(data, 'height')
    loc_dir = kf.apply_kalman_filter(data, 'direction')
    loc_hor = kf.apply_kalman_filter(data, 'horizontal_accuracy')
    loc_ver = kf.apply_kalman_filter(data, 'vertical_accuracy')

elif (measurement == 'Magnetometer'):
    # combined_measurement.rename(columns={'Magnetic field x (ÂµT)': 'x', 'Magnetic field y (ÂµT)': 'y', 'Magnetic field z (ÂµT)': 'z'}, inplace=True)
    mag_x = kf.apply_kalman_filter(data, 'x')
    mag_y = kf.apply_kalman_filter(data, 'y')
    mag_z = kf.apply_kalman_filter(data, 'z')









