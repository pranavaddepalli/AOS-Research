#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:37:03 2019

@author: pranavaddepalli
"""
#%% Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


np.set_printoptions(precision=3, suppress=True)

#%% Preprocessing
def preprocess(raw_data, infill):
    temperatures = []
    
    for a in raw_data:
        for i in a:
            temperatures.append(i)
    temperatures = np.asarray(temperatures)
    
    print("Total number of data points / temperature values is ",len(temperatures))
    
    data = np.empty(shape=(len(raw_data)*len(raw_data[0]),4))
    #col 0 is X
    #col 1 is Y
    #col 2 is time
    #each row is for every measurement that happened
    
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
            
            data[r][0] = x
            data[r][1] = y
            data[r][2] = r
            data[r][3] = infill
            r += 1
    print("Finished preprocessing")
    return(data, temperatures)


#%% Creating dataset and splitting
ten_percent = np.delete(np.genfromtxt('Pranav_10percentLines.txt', delimiter=','), [2, 3, 6], 1)
twenty_percent = np.genfromtxt('20pLines.csv', delimiter=',')[:,:-1]


'''
normalized:
    10 is 0
    20 is 0.25
    30 is 0.5
    40 is 0.75
    50 is 1
'''

ten_data,ten_temps = preprocess(ten_percent, 10)

twenty_data, twenty_temps = preprocess(twenty_percent, 20)

data = np.concatenate((ten_data, twenty_data))
temperatures = np.concatenate((ten_temps, twenty_temps))
x_train, x_test, y_train, y_test = train_test_split(data, temperatures, test_size=0.2, random_state=42)

print("Split into training and testing - 0.2 test size")







#%% Model Architecture
cp_callback = tf.keras.callbacks.ModelCheckpoint("cp.ckpt", 
                                                 verbose=1,
                                                 period=10)

model = tf.keras.Sequential()
#input
model.add(layers.Dense(4, input_dim=4))
model.add(layers.LeakyReLU(alpha=0.05))
#hidden layer 1
model.add(layers.Dense(64))
model.add(layers.LeakyReLU(alpha=0.05))
#hidden layer 2
model.add(layers.Dense(32))
model.add(layers.LeakyReLU(alpha=0.05))
#hidden layer 3
model.add(layers.Dense(4))
model.add(layers.LeakyReLU(alpha=0.05))
#output layer
model.add(layers.Dense(1))
model.add(layers.LeakyReLU(alpha=0.05))


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), #adam optimizer
              loss='mse', #mean squared error
              metrics=['mae']) #mean absolute error

print("Model built!")


#%% Training
model.fit(x=x_train,
          y=y_train, 
          batch_size=32, 
          epochs=72, 
          validation_data= (x_test, y_test),
          callbacks = [cp_callback])
#%% Save
model.save('model.h5')
