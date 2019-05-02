#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:38:35 2019

@author: pranavaddepalli
"""
#%% Imports
import pickle
import pandas as pd
import numpy as np
import math
import time


#%% Recreate dataset
df = pd.read_pickle("10_percent_dataframe.pickle")


#%% Calculate sigma
'''
Sigma represents the temperature distribution in the 3D space. There was no convention so I chose this myself.
Sigma0 refers to a value of 0, or even, temperture distribution. This is found by using a constant value for the
temperatures and finding the centroid of the entire set of points. This is essentially a 2-dimensional centroid.
Sigma0 is then used with calculated values of the new centroid of each set of points at time t to construct a
vector, which we will refer to as Sigma, that shows the magnitude and direction of temperature flow.

Here, we calculate Sigma0 based on the set of positions, and then calculate the vector Sigma.

'''
#Our points
x_positions = [70, 55, 32.4, 24, 0, -22.5, -45, -72.5]
y_positions = [-70, 56, -41.8, 15, -12, 33, -48, 56]

#Function to return centroid as a tuple
def centroid(x, y, t):
    return (np.mean(x), np.mean(y), np.mean(t))

sigma0 = centroid(x_positions, y_positions, 0)
print("Sigma0 is at the point {} ".format(sigma0, decimals=5))

vectors = {}

print("Calculating vectors...")
start = time.time()
#Calculate the vector for each second
for t in range(40000, 40001):
    temperatures = df.iloc[t].values.tolist()
    
    c = centroid(x_positions, y_positions, temperatures)
    x1,y1,x2,y2 = sigma0[0],sigma0[1],c[0],c[1]
    print(c)
    

    m = (y2 - y1) / (x2 - x1)
    angle = np.arctan(m)
    distance = (((x2-x1)**2)+((y2-y1)**2))**(1/2)
    
    vectors[t] = (distance, angle)

    
print("Finished calculation in {} seconds".format(time.time() - start))



