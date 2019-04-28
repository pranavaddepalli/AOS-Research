#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:02:18 2019

@author: pranavaddepalli
"""
import pandas as pd
import time
from scipy import stats

'''
Creating and cleaning temperature dataset and running an ANOVA test 
to find if the temperatures are significantly different across the 
thermistors.
'''
#read in data as Pandas dataframe
df = pd.read_csv("Pranav_10percentLines.csv")

#printing initial format of dataframe
print("Shape of dataset: ", df.shape)

#replace negative values (not possible) with 0
df[df < 0] = 0

#drop data from useless thermistors (optional)
df = df.drop(df.columns[2], axis=1)
df = df.drop(df.columns[2], axis=1)
df = df.drop(df.columns[4], axis=1)
print("Cut columns and replaced negatives")

#print new format of dataframe
print("New shape of dataset: ", df.shape)


#analyze data for existence of distribution
statistic, p_value = stats.f_oneway(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]], df[df.columns[3]], df[df.columns[4]],)
print("Distribution ANOVA test")
print("Statistic: ", statistic)
print("P-value: ", p_value)


'''
Gradient calculation for each of the points
'''
start = time.time()
print("Calculating gradients...")
#Gradient calculation function between points in 3D
def gradient(point_one, point_two):
    x1, y1, t1 = point_one[0], point_one[1], point_one[2]
    x2, y2, t2 = point_two[0], point_two[1], point_two[2]

    dt = t2 - t1
    dx = (((x2-x1)**2)+((y2-y1)**2)+((t2-t1)**2))**(1/2)
    
    return dt/dx

x_positions = [1, 2, 3, 4, 5]
y_positions = [1, 2, 3, 4, 5]

gradients = {}


#Calculate all gradients

for t in range(0, len(df[df.columns[0]])):
    gradients_at_time = []

    #List of temperatures at time t
    temperatures = df.iloc[t].values.tolist()

    #List of points at time t
    points = []
    for x, y, temp in list(zip(x_positions, y_positions, temperatures)):
        points.append([x, y, temp])
    
    #gradient calculation
    for point in points:
        for next_point in points[1:] + [points[0]]:
            if point != next_point:
                gradients_at_time.append(gradient(point, next_point))
    
    
    gradients[t] = gradients_at_time

end = time.time()
print("Finished calculation in ", (end - start), "seconds.")
print("Type gradients[t] for the gradients at any time t.")
            