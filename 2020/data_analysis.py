# -*- coding: utf-8 -*-
# SETUP and READ DATA

import numpy as np

# 0.2 layer height, 0.04 wall thickness, 10%, lines
raw_data_A = np.genfromtxt('Data/DATA_.2lh_.04wt_10p_lines_DATA.txt', delimiter=',')
print("A: {} columns and {} rows".format(np.size(raw_data_A, axis=1), np.size(raw_data_A, axis=0)))

# 0.2 layer height, 0.04 wall thickness, 10%, tri-hexagonal
raw_data_B = np.genfromtxt('Data/DATA_.2lh_.04wt_10p_trihex_DATA.txt', delimiter=',')
print("B: {} columns and {} rows".format(np.size(raw_data_B, axis=1), np.size(raw_data_B, axis=0)))

# 0.2 layer height, 0.04 wall thickness, 20%, lines
raw_data_C = np.genfromtxt('Data/DATA_.2lh_.04wt_20p_lines_DATA.txt', delimiter=',')
print("C: {} columns and {} rows".format(np.size(raw_data_C, axis=1), np.size(raw_data_C, axis=0)))

print("{} total data points".format( (np.size(raw_data_A, axis=1) * np.size(raw_data_A, axis=0)) + (np.size(raw_data_B, axis=1) * np.size(raw_data_B, axis=0)) + (np.size(raw_data_C, axis=1) * np.size(raw_data_C, axis=0))    ))

data = {
        0.1: {
                "10%":{
                        "lines":{
                                .04: [],
                                .08: []
                                },
                        "trihex":{
                                .04: [],
                                .08: []                                
                                }
                        },
                "20%":{
                        "lines":{
                                .04: [],
                                .08: []                                
                                },
                        "trihex":{
                                .04: [],
                                .08: []                                
                                }                        
                        },
                "30%":{
                        "lines":{
                                .04: [],
                                .08: []                                
                                },
                        "trihex":{
                                .04: [],
                                .08: []                                
                                }                        
                        }
                },
        0.2: {
                "10%":{
                        "lines":{
                                .04: [],
                                .08: []                                
                                },
                        "trihex":{
                                .04: [],
                                .08: []                                
                                }                        
                        },
                "20%":{
                        "lines":{
                                .04: [],
                                .08: []                                
                                },
                        "trihex":{
                                .04: [],
                                .08: []                               
                                }                        
                        },
                "30%":{
                        "lines":{
                                .04: [],
                                .08: []                                
                                },
                        "trihex":{
                                .04: [],
                                .08: []                                
                                }                        
                        }                
                }
        }

def gen_point(raw_data_disc, layer_height, density, pattern, wall_thickness, row, col):
    temperature = raw_data_disc[row, col]
    time = row
    
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
        
    return [layer_height, density, pattern, wall_thickness, time, x, y, temperature]

def get_center(points):
    x_values = []
    y_values = []
    temps = []
    
# %% CLEAN DATA
#to fix disc B data using mean change
for i in range(np.where(raw_data_B == -273.15)[0][0], int(raw_data_B.size / raw_data_B[0].size)):
    raw_data_B[i][0] = raw_data_B[i - 1][0] + np.mean(raw_data_B[i - 1][1:7] - raw_data_B[i - 2][1:7])
print('B data cleaned')

# %% GENERATE POINTS

#disc A -- 0.2 LH, 10%, Lines, 0.04 WT
temp = []
for row in range(0, np.size(raw_data_A, axis=0)):
    row_temp = []
    for col in range(0, raw_data_A[0].size):
        row_temp.append(gen_point(raw_data_A, 0.2, '10%', 'lines', 0.04, row, col))
    temp.append(row_temp) 

data[0.2]['10%']['lines'][0.04] = temp
print('A data stored and points generated')

#disc B -- 0.2 LH, 10%, Trihex, 0.04 WT
temp = []
for row in range(0, np.size(raw_data_B, axis=0)):
    row_temp = []
    for col in range(0, raw_data_B[0].size):
        row_temp.append(gen_point(raw_data_B, 0.2, '10%', 'trihex', 0.04, row, col))
    temp.append(row_temp) 

data[0.2]['10%']['trihex'][0.04] = temp
print('B data stored and points generated')

#disc C -- 0.2 LH, 20%, Lines, 0.04 WT
temp = []
for row in range(0, np.size(raw_data_C, axis=0)):
    row_temp = []
    for col in range(0, raw_data_C[0].size):
        row_temp.append(gen_point(raw_data_C, 0.2, '20%', 'lines', 0.04, row, col))
    temp.append(row_temp) 

data[0.2]['20%']['lines'][0.04] = temp
print('C data stored and points generated')

# %% THERMAL EQUILIBRIUM CALCULATIONS (1% of data)

#disc A
temp = []
for t in range(0, int(0.01 * np.size(data[0.2]['10%']['lines'][0.04], axis=0))):
    temp.append(gen_center(data[0.2]['10%']['lines'][0.04][t]))