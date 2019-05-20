#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:37:03 2019

@author: pranavaddepalli
"""
#%% Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
print("""Available devices:
    Device 0: {} | {} | Available Memory: {}
    Device 1: {} | {} | Available Memory: {}
    """.format(devices[0].device_type, 
    devices[0].physical_device_desc, 
    devices[0].memory_limit, 
    devices[1].device_type, 
    devices[1].physical_device_desc, 
    devices[1].memory_limit))

#%% Preprocessing
def preprocess(raw_data, infill):
    '''
    Organizes the data into:
    
    DATA
    col 0 is time
    col 1 is X position of thermistor
    col 2 is Y position of thermistor
    col 3 is infill percentage

    TEMPERATURES
    col 0 is the temperature
    '''
    print("Raw data has shape {}".format(raw_data.shape))
    data = np.empty(shape=(len(raw_data)*len(raw_data[0]),4))
    temperatures = []
    
    r = 0
    for col in range(0, np.size(raw_data, 1)):
        for row in range(0, np.size(raw_data, 0)):
            if col == 0:
                 x = 70
                 y = -70
            elif col == 1:
                x = 55
                y = 56
            elif col == 2:
                x = 32.4
                y = -41.8
            elif col ==3:
                x = 24
                y = 15
            elif col == 4:
                x = 0
                y = -12
            elif col == 5:
                x = -22.5
                y = 33
            elif col == 6:
                x = -45
                y = 48
            else:
                x = -72.5
                y = 56
            
            data[r][0] = r
            data[r][1] = x
            data[r][2] = y
            data[r][3] = infill
            temperatures.append(raw_data[row, col])
            
            r += 1
    print("Finished preprocessing for infill", infill)

    #fix problem with 10% dataset
    
    if infill == 0:
        print("Fixing infill 10% data", )
        indices = [i for (i,v) in enumerate(temperatures) if v==385.24]
        for i in indices:
            temperatures.pop(i)
            data = np.delete(data, (i), axis=0)
        print("Fixed 10% infill data")
    return(data, temperatures)


#%% Creating dataset and splitting

try:
    temperatures=np.load('temperatures.npy')
    data=np.load('data.npy')
    print("loaded from file")
except FileNotFoundError:
    with tf.device('cpu:0'):
        ten_data, ten_temperatures = preprocess(np.genfromtxt('Pranav_10percentLines.txt', delimiter=','), 0)
        twenty_data, twenty_temperatures = preprocess(np.genfromtxt('20pLines.csv', delimiter=',')[:,:-1], .2)
    
    data = np.concatenate((ten_data, twenty_data))
    temperatures = np.concatenate((ten_temperatures, twenty_temperatures))
    np.save('data', data)
    np.save('temperatures', temperatures)
    print("Saved dataset to data.npy and temperatures.npy")


x_train, x_test, y_train, y_test = train_test_split(data, temperatures, test_size=0.2, random_state=42)

print("Split into training and testing - 0.2 test size")






#%% Model Architecture

cp_callback = tf.keras.callbacks.ModelCheckpoint("cp.ckpt", 
                                                 verbose=1,
                                                 period=10)
model = tf.keras.Sequential()
#input
model.add(keras.layers.Conv2D(1, (10,10), input_shape=(2,442184,7,7)))
#hidden layer 1
model.add(keras.layers.Dense(32))
model.add(keras.layers.LeakyReLU(alpha=0.05))
#hidden layer 2
model.add(keras.layers.Dense(64))
model.add(keras.layers.LeakyReLU(alpha=0.05))
#hidden layer 3
model.add(keras.layers.Dense(32))
model.add(keras.layers.LeakyReLU(alpha=0.05))
#hidden layer 4
model.add(keras.layers.Dense(4))
model.add(keras.layers.LeakyReLU(alpha=0.05))
#output layer
model.add(keras.layers.Dense(1, activation='linear'))


def rmse(y_true, y_pred):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))
    
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), #adam optimizer
              loss='mse', #mean squared error
              metrics=['mape', 'mae', rmse]) #mean absolute error
 
print("Model built!")

#%% Training
with tf.device('gpu:0'):
    history = (model.fit(x=x_train,
              y=y_train, 
              batch_size=64, 
              epochs=1000, 
              validation_data= (x_test, y_test),
              callbacks = [cp_callback]))

#%% Save
model.save('model.h5')

#%% Plot

