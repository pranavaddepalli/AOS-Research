# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:59:39 2019

@author: pranavaddepalli
"""
#%% SETUP and LOAD RAW DATA
import numpy as np 
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

np.set_printoptions(precision=3, suppress=True)

base_dir = os.getcwd()
data_dir = base_dir + "\Data\\"
image_dir = base_dir + "\Images\\"
print("Working Directory: {}. \nData Directory: {}".format(base_dir, data_dir))

raw_10_percent = np.genfromtxt(data_dir + '10pLines', delimiter=',')
print("10% infill data has {} columns and {} rows.".format(np.size(raw_10_percent, axis=1), np.size(raw_10_percent, axis=0)))

raw_20_percent = np.genfromtxt(data_dir + '20pLines', delimiter=',')[:, :8]
print("20% infill data has {} columns and {} rows.".format(np.size(raw_20_percent, axis=1), np.size(raw_20_percent, axis=0)))

raw_30_percent = np.genfromtxt(data_dir + '30pLines', delimiter=',')
print("30% infill data has {} columns and {} rows.".format(np.size(raw_30_percent, axis=1), np.size(raw_30_percent, axis=0)))

#%% PROCESS RAW DATA

# Create dataset with points
def point(raw, row, col):
    n = col
    temperature = raw[row, col]
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
    return n, x, y, temperature

# shape of data: (infill, time, thermistor) ==> x,y,temperature
system = np.zeros(shape = (3, len(raw_30_percent) + len(raw_20_percent) + len(raw_10_percent), 8), dtype='O')

for i in range(0,30, 10):
    infill = globals()['raw_' + str(i + 10) + '_percent']
    print("Creating points for {}% infill...".format(i + 10), end="", flush=True)
    for col in range(0, np.size(infill, axis=1)):
        for row in range(0, np.size(infill, axis=0)):
            n, x, y, temperature = point(infill, row, col)
            system[int(i / 10), row, n] = (x, y, temperature)
    print("Done!")

#%% HEAT MAP GENERATION



#%% CENTER CALCULATIONS

#center of mass calculation assuming linearity and that mass = temperature
# (since heat flow in is directly proportional to temperature change dT/dx)

# weighted mean temp is calculated using a formula that doesn't really exist but is kept constant. 
# sum of all (temperature / distance from equilibrium)
    
    
x_graph_list = [[] for _ in range(3)]
y_graph_list = [[] for _ in range(3)]
equilibrium = (5.175, 10.525, 0)
def center(points):
    global equilibrium
    global x_graph_list
    global y_graph_list
    weighted_mean_temp = sum([point[2] for point in points]) / 8
    #weighted_mean_temp = sum((point[2] / ((point[0] - equilibrium[0])**2) + ((point[1] - equilibrium[0])**2)) for point in points)
    x = sum([point[0] * point[2] for point in points]) / sum([point[2] for point in points])
    y = sum([point[1] * point[2] for point in points]) / sum([point[2] for point in points])
    return (x, y, weighted_mean_temp)


centers = [[] for _ in range(3)]
for infill in range(0, 3):
    print("Calculating centers for {}% infill...".format((infill + 1)*10), end="", flush=True)
    for time in system[infill]:
        if type(time[0]) is tuple:
            x, y, temperature_mean = center(time)
            x_graph_list[infill].append(x)
            y_graph_list[infill].append(y)
            centers[infill].append((x, y, temperature_mean))
        else: break
    print("Done!")
#temperature gradient = (partial derivative of T / dx, partial derivative of T / dy)
        
#%% VECTOR CALCULATIONS
gradients = [[] for _ in range(3)]
for infill in range(0, 3):
    print("Calculating gradients for {}% infill...".format((infill + 1)*10), end="", flush=True)
    for c in centers[infill]:
        dx = c[0] - equilibrium[0]
        dy = c[1] - equilibrium[1]
        dt = c[2] - equilibrium[2]
        gradT = np.sqrt( ((dt / dx)**2) + ((dt / dy)**2) )
        direction = np.degrees(np.arctan(dy / dx))
        gradients[infill].append((dx, dy, dt, gradT, direction))
    print("Done!")
    
    
#%% VISUALIZING GRADIENTS
fig = plt.figure()

#STATIC VECTOR FIELD
'''
plt.subplot(221)
plt.plot(equilibrium[0], equilibrium[1], "or")
plt.xlim(-0, 15)
plt.ylim(-0, 15)
plt.scatter(x_graph_list[1], y_graph_list[1], s=1)
'''
#ANIMATION
plt.subplot(111)
plt.plot(equilibrium[0], equilibrium[1], "or", label="Equilibrium Point")
plt.xlim(2.5, 15)
plt.ylim(2.5, 15)
plt.ylabel("Y position (mm)")
plt.xlabel("X position (mm)")
plt.title("20% infill animation: Movement of center of temperature over time")
graph = plt.scatter([], [])
def animate(i):
    if i > len(x_graph_list[1]) - 1:
        i = len(x_graph_list[1]) -1
    graph.set_offsets(np.vstack((x_graph_list[1][:i+1], y_graph_list[1][:i+1])).T)
    graph.set_sizes(np.ones(len(x_graph_list)))
    return graph
ani = FuncAnimation(fig, animate, frames=len(x_graph_list[1]), interval=.00001)

plt.rcParams['animation.ffmpeg_path'] = 'F:\Program Files\FFmpeg\ffmpeg-20190518-c61d16c-win64-static\bin'

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save('20_percent_animation.mp4', writer=writer)
plt.show()


'''
ax = plt.gca()
ax.scatter(x_graph_list[1], y_graph_list[1], s=1)
ax.plot(equilibrium[0], equilibrium[1], "or")

#soa = np.array([[equilibrium[0], equilibrium[1], gradients[0][0][0], gradients[0][0][1]]])
#X, Y, U, V = zip(*soa)
#ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)


ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
'''


plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    