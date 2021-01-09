import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)


def data_smooth(input_data):
    
    # make the data curve smooth    
    
    # define the differnece sequence
    difference = input_data.diff().dropna()
    
    # define description of the sequence
    information = difference.describe()
    
    # define maximum normal value
    high_value = information['75%'] * 0.75
    
    # define minimum normal value
    low_value = information['25%'] * 1.5
    
    # define index of abnormal values
    abnormal_index = difference[(difference > high_value) | (difference < low_value)].index
    
    i = 0 
    data = np.array(input_data)
    while i < len(abnormal_index):
        
        # find how many continuous abnormal values
        n = 1
        
        # the starting index of the abnormal values
        start = abnormal_index[i]
        if (i+n) < len(abnormal_index):   
            
            while abnormal_index[i+n] == (start + n):
                n += 1
                
                if (i+n) >= len(abnormal_index):
                    break                
        i += n - 1
        
        # the ending index of the abnormal values
        end = abnormal_index[i]
        
        # fill abnormal values with appropriate values
        padding = input_data[start - 1]
        value = np.linspace(padding, padding, n)
        data[start:end+1] = value
        i += 1

    return data


def data_normalization(input_data):
    
    # normalize the input data
    
    # create normalization data list
    new_data = []
    
    # define description of the input data
    information = input_data.describe()
    
    # new_data = (old_data - old_data_min)/(old_data_max - old_data_min)
    for i in input_data:
         new_data.append((i - information['min']) / (information['max'] - information['min']))
    
    return np.array(new_data)