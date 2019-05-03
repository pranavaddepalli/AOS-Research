#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:37:03 2019

@author: pranavaddepalli
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)



#fixing the dataset
raw_data = np.genfromtxt('Pranav_10percentLines.csv', delimiter=',')
print("Raw data shape is ", raw_data.shape)

temperatures = []

for a in raw_data:
    for i in a:
        temperatures.append(i)
temperatures = np.asarray(temperatures)

temperatures[0] = 49.58
print("Total number of data points / temperature values is ",len(temperatures))

data = np.empty(shape=(282255,3))
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
        r += 1


print("Finished pre-processing")


x_train, x_test, y_train, y_test = train_test_split(data, temperatures, test_size=0.2, random_state=42)

print("Split into training and testing - 0.2 test size")

model = tf.keras.Sequential()
#input
model.add(layers.Dense(4, input_dim=3, activation='relu'))
#hidden layer 1
model.add(layers.Dense(12, activation='relu'))
#hidden layer 2
model.add(layers.Dense(12, activation='relu'))
#output layer
model.add(layers.Dense(1, activation='relu'))

model.compile(optimizer=tf.train.AdamOptimizer(0.00001), #gradient descent
              loss='mse', #mean squared error
              metrics=['mae']) #mean absolute error

print("Model built!")
print(model.summary())


model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

