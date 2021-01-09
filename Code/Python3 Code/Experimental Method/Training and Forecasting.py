import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)


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


def list_simple_transform_csv(Elevation):

    # transform list into csv
    
    # define name of sequences
    column = Elevation
    name =['Elevation']

    # define csv
    combine_list = []
    
    for i in range(len(column)):
        combine_list.append([column[i]])
    
    data_csv = pd.DataFrame(columns = name, data = combine_list)

    return data_csv


def create_sample_and_label(input_data, number, set_index, slice1, slice2, slice3, slice4, choice):
    
    # create training samples and sample labels
    
    # read normalized data and transform it into csv
    data_nor = data_normalization(input_data)[:number]
    csv = list_simple_transform_csv(data_nor)
    
    # define data list and csv index
    data_list = data_nor.tolist()
    index = csv.index.tolist()
    
    # define training samples
    training_sample = []
    
    for i in index :
        data_slice = data_list[i:i + slice1]
        
        if choice == 'true':
            for j in range(set_index):
                data_slice.append(1e-3)
                
        training_sample.append(data_slice)
        
        if i + slice2 == len(data_list) - 1:
            break
    
    # define sample labels
    sample_label = []
    
    for i in index:
        sample_label.append([data_list[i + slice3]])
        
        if i + slice4 == len(data_list)-1:
            break
            
    return training_sample, sample_label


def prediction_historical_sample(input_data1, input_data2, number, index):
    
    # create input historical samples
    
    data_nor1 = data_normalization(input_data1)[:number]
    data_nor2 = data_normalization(input_data2)[:number]
    
    new_data = (data_nor1 + data_nor2) / 2
    
    return new_data[:index].tolist()


def iteration_prediction_list(neural_network,historical_sample1, iterations, steps, number, choice):
    
    # use historical samples to do prediction
    
    for i in range(0, iterations):
        historical_sample = historical_sample1
        input_data = historical_sample[i:i+steps]
        
        if choice == 'true':
            for i in range(number):
                input_data.append(1e-3)
                
        prediction_data = neural_network.predict(input_data)
        historical_sample.append(prediction_data[0])
    
    return historical_sample, len(historical_sample)


def data_transfer(input_data, normalized_data):
    
    # transfer normalized data into actual data
    
    # create actual data list
    new_data = []
    
    # define description of the input data
    information = input_data.describe()
    
    # new_data = (old_data * (old_data_max - old_data_min) + old_data_min)
    for i in normalized_data:
        new_data.append(i * (information['max'] - information['min']) + information['min'])
    
    return np.array(new_data), len(new_data)