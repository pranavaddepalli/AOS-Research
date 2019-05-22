# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:59:39 2019
@author: pranavaddepalli
"""
#%% SETUP and LOAD RAW DATA
import numpy as np 
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

np.set_printoptions(precision=3, suppress=True)

base_dir = os.getcwd()
data_dir = base_dir + "/Data/"
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

# fixing data for 10% infill
#raw_10_percent = np.delete(raw_10_percent,[2,3,6],1)

# shape of data: (infill, time, thermistor) ==> x,y,temperature
system = np.zeros(shape = (3, len(raw_30_percent) + len(raw_20_percent) + len(raw_10_percent), 8), dtype='O')

hm_x = [[] for _ in range(3)] 
hm_y = [[] for _ in range(3)]
hm_t = [[] for _ in range(3)]
hm_temp = [[] for _ in range(3)]

for i in range(0,30, 10):
    infill = globals()['raw_' + str(i + 10) + '_percent']
    print("Creating points for {}% infill...".format(i + 10), end="", flush=True)
    for col in range(0, np.size(infill, axis=1)):
        for row in range(0, np.size(infill, axis=0)):
            n, x, y, temperature = point(infill, row, col)
            hm_x[int(i / 10)].append(x)
            hm_y[int(i / 10)].append(y)
            hm_t[int(i / 10)].append(row)
            hm_temp[int(i / 10)].append(temperature)
            system[int(i / 10), row, n] = (x, y, temperature)
    print("Done!")

#%% graphics for poster



#%% HEAT MAP GENERATION
##HEAT MAP
'''
from matplotlib import cm
plt.style.use('classic')

hm20 = plt.figure()
ax = hm20.gca(projection='3d')
ax.scatter(hm_x[1], hm_y[1], hm_t[1], c=hm_temp[1], lw=0, s=10)

asdf, jkl = np.meshgrid(hm_x[1], hm_y[1], sparse=True)
fdsa, lkj = np.meshgrid(hm_t[1], hm_temp[1], sparse=True)
for t in range(0, len(hm_t[1])):
    surf = ax.plot_surface(np.asarray(asdf), np.asarray(jkl), np.asarray(lkj))

hm_x_twenty = [[] for _ in range(len(system[0]))] 
hm_y_twenty = [[] for _ in range(len(system[0]))] 
hm_t_twenty = [[] for _ in range(len(system[0]))] 
hm_temp_twenty = [[] for _ in range(len(system[0]))] 

for t in range(0, len(system[0])):
    for p in system[1][t]:
        hm_x_twenty[t].append(p[0])
        hm_y_twenty[t].append(p[1])
        hm_t_twenty[t].append(t)
        hm_temp_twenty[t].append(p[2])

for t in range(0, len(hm_t_twenty)):
    surf = ax.plot_trisurf(hm_x_twenty, hm_y_twenty, hm_t_twenty)
    
cmap = cm.get_cmap('PiYG', 11)
img = ax.scatter(hm_x[1], hm_y[1], hm_t[1], c=hm_temp[1], cmap=cmap)
cbar = hm20.colorbar(img, boundaries=[0, 10, 20, 30])
'''



#%% CENTER CALCULATIONS

#center of mass calculation assuming linearity and that mass = temperature
# (since heat flow in is directly proportional to temperature change dT/dx)

# weighted mean temp is calculated using a formula that doesn't really exist but is kept constant. 
# sum of all (temperature / distance from equilibrium)
    
    
x_graph_list = [[] for i in range(3)]
y_graph_list = [[] for i in range(3)]
avgTemp_graph_list = [[] for i in range(3)]
equilibrium = (5.175, 10.525, 0)
def center(points):
    global equilibrium
    global x_graph_list
    global y_graph_list
    weighted_mean_temp = sum((((((p[0])**2) + 
                                ((p[1])**2))**0.5) 
                                * p[2]) for p in points)    
    tmpKieran = (sum(((((point[0])**2) + ((point[1])**2))**0.5) for point in points))
    weighted_mean_temp = weighted_mean_temp / tmpKieran
    
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
            avgTemp_graph_list[infill].append(temperature_mean)
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
#%%VISUALIZATION
#VECTORS

#origin = [equilibrium[0]], [equilibrium[1] ]
vectors = np.array([list(a) for a in zip(x_graph_list[1], y_graph_list[1])])
origin = [0], [0]
soa = np.array([[equilibrium[0], equilibrium[1], x_graph_list[1][c_], y_graph_list[1][c_]] for c_ in range(0, len(x_graph_list[1]))])
X, Y, U, V = zip(*soa)

plt.plot(equilibrium[0], equilibrium[1], "or")
plt.scatter(x_graph_list[1], y_graph_list[1], s=1)

plt.quiver(X, Y, U, V, scale=1)
plt.draw()
plt.show()




#%% centers
# CENTERS
figs = []

for GRAPHING_INFILL in range(0, 3) :
    
    figs.append(plt.figure())
    
    plt.subplot(111)
    ax = plt.gca()
    ax.scatter(x_graph_list[GRAPHING_INFILL], y_graph_list[GRAPHING_INFILL], s=1)
    ax.plot(equilibrium[0], equilibrium[1], "or")
    plt.ylabel("Y position (mm)")
    plt.xlabel("X position (mm)")
    plt.title("Temperature Centers for {}% infill".format(format((GRAPHING_INFILL + 1)*10)))
    


plt.show()

#soa = np.array([[equilibrium[0], equilibrium[1], gradients[0][0][0], gradients[0][0][1]]])
#X, Y, U, V = zip(*soa)
#ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)




# animation
'''
anim = plt.figure()
plt.subplot(111)
plt.plot(equilibrium[0], equilibrium[1], "or", label="Equilibrium Point")
plt.ylabel("Y position (mm)")
plt.xlabel("X position (mm)")
plt.title("20% infill animation: Movement of center of temperature over time")
graph = plt.scatter([], [])
def animate(i):
    if i > len(x_graph_list[GRAPHING_INFILL]) - 1:
        i = len(x_graph_list[GRAPHING_INFILL]) -1
    graph.set_offsets(np.vstack((x_graph_list[GRAPHING_INFILL][:i+1], y_graph_list[GRAPHING_INFILL][:i+1])).T)
    graph.set_sizes(np.ones(len(x_graph_list)))
    return graph
ani = FuncAnimation(anim, animate, frames=len(x_graph_list[GRAPHING_INFILL]), interval=.00000001)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('20_percent_animation.mp4', writer=writer)
'''


#%% STATISTICS
import scipy.stats as stats

#STANDARD DEVIATION OF CENTERS

print("Standard Deviation of Temperature Centers:")

x_std_ten = np.std(x_graph_list[0])
y_std_ten = np.std(y_graph_list[0])
print("\nTen percent: \n")
print("STD in X: {} \nSTD in Y: {}".format(x_std_ten,y_std_ten))

x_std_twenty = np.std(x_graph_list[1])
y_std_twenty = np.std(y_graph_list[1])
print("\nTwenty percent:\n")
print("STD in X: {} \nSTD in Y: {}".format(x_std_twenty,y_std_twenty))

x_std_thirty = np.std(x_graph_list[2])
y_std_thirty = np.std(y_graph_list[2])
print("\nThirty percent: \n")
print("STD in X: {} \nSTD in Y: {}".format(x_std_thirty,y_std_thirty))


##GRADIENTS
ten_gradients = [value[3] for value in gradients[0] ]
twenty_gradients = [value[3] for value in gradients[1] ]
thirty_gradients = [value[3] for value in gradients[2] ]
anova_statistic, anova_pvalue = stats.f_oneway(ten_gradients, twenty_gradients, thirty_gradients)
statistic_20_30, pvalue_20_30 = stats.ttest_ind(twenty_gradients, thirty_gradients, equal_var=False)

print("\nANOVA TEST:\n-------------------- \nStatistic: {} \np-value: {}\n".format(anova_statistic, anova_pvalue))
print("Two-Sample T Test for Independence with unequal variances:\n--------------------")
#print("10% to 30%:\n--------------------\nStatistic: {} \np-value: {}".format(statistic_10_20, pvalue_10_20))
print("20% to 30%:\n--------------------\nStatistic: {} \np-value: {}".format(statistic_20_30, pvalue_20_30 ))
#%% ML MODEL



#%% PRINTING DATA
import pandas as pd
thirtydf = pd.DataFrame(gradients[2], columns =['X', 'Y', 'Mean Temperature', 'Gradient from Equilibrium', 'Angle'], dtype = float) 

writer = pd.ExcelWriter("thirtydf.xlsx")
thirtydf.to_excel(writer, 'Sheet1')
writer.save()

#pd.DataFrame(list(zip(x_graph_list[1], y_graph_list[1], avgTemp_graph_list[1])), columns =['X', 'Y', 'Temperature'])
