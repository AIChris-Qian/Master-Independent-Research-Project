import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)


def read_data(filename):
    
    # read data from files into list
    
    # create three sequences
    index = []
    hour = []
    elevation = []
    
    # open file
    infile = open(filename, "r")
    
    for line in infile:
        numbers = line.split()
        
        if len(numbers) !=5:
            continue
            
        cycle = float(numbers[0])
        height = float(numbers[3])
    
        index.append(cycle)
        hour.append((cycle - 1)/ 4)
        elevation.append(height)
    
    return index, hour, elevation


def list_transform_csv(Index, Hour, Elevation, scale, number):    
    
    # transform list into csv    
    
    # slice three sequences according to timescale
    column_1 = Index[scale:number:scale]
    column_2 = Hour[scale:number:scale]
    column_3 = Elevation[scale:number:scale]
    
    # define name of sequences
    name =['Cycle', 'Hour', 'Elevation']
    
    # define csv
    combine_list = []
    
    for i in range(len(column_1)):
        combine_list.append([column_1[i], column_2[i], column_3[i]])
    
    data_csv = pd.DataFrame(columns = name, data = combine_list)

    return data_csv


def onefigure_plot(start, end, x_axis, y_axis, colour, label_name, title_name, True_or_False):
    
    # plot one figure
    
    fig, ax1 = plt.subplots(1, figsize=(15, 5))
    fig.tight_layout(w_pad=4)
    
    # draw a graph of one data  
    ax1.plot(np.array(x_axis), np.array(y_axis), colour, label = label_name, markersize=5)

    # set legend information
    ax1.set_ylim([start, end]);
    ax1.set_xlabel('Hours (h)', fontsize=16)
    ax1.set_ylabel('Tidal elevation (m)', fontsize=16)
    ax1.set_title(title_name, fontsize=16)
    ax1.legend(loc='best', fontsize=14)
    
    # add grid or not
    ax1.grid(True_or_False)
    
    return 'one figure plot'


def twofigure_plot(start, end, x_axis1, y_axis1, colour1, label_name1, x_axis2, y_axis2, colour2, label_name2, title_name, True_or_False):
    
    # plot two figures  
    
    fig, ax1 = plt.subplots(1, figsize=(15, 5))
    fig.tight_layout(w_pad=4)
    
    # draw a graph of two kinds of data 
    ax1.plot(np.array(x_axis1), np.array(y_axis1), colour1, label = label_name1, markersize=5)
    ax1.plot(np.array(x_axis2), np.array(y_axis2), colour2, label = label_name2, markersize=5)
    
    # set legend information
    ax1.set_ylim([start, end]);
    ax1.set_xlabel('Hours (h)', fontsize=16)
    ax1.set_ylabel('Tidal elevation (m)', fontsize=16)
    ax1.set_title(title_name, fontsize=16)
    ax1.legend(loc='best', fontsize=14)

    # add grid or not
    ax1.grid(True_or_False)
    
    return 'two figures plot'