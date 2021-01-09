import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)


def correlation_coefficient(prediction, observation):
    
    # calculate correlation coefficient regarding prediction and observation
    
    # define mean of prediction and observation
    prediction_csv = list_simple_transform_csv(prediction)
    prediction_mean = prediction_csv['Elevation'].describe()['mean']
    observation_mean = observation.describe()['mean']
    
    # define cov(x,y)
    covariance_xy = 0
    for i in range(0, len(prediction_csv)):
        covariance_xy += (prediction_csv['Elevation'][i] - prediction_mean) * (observation[i] - observation_mean)
    
    # define var[x]
    variance_x = 0
    for i in range(0, len(prediction_csv)):
        variance_x += (prediction_csv['Elevation'][i] - prediction_mean)**2
    
    # define var[y]
    variance_y = 0
    for i in range(0, len(prediction_csv)):
        variance_y += (observation[i] - observation_mean)**2
    
    # r(x,y) = cov(x,y) / sqrt(var[x] * var[y])
    correlation_coefficient = covariance_xy / np.sqrt(variance_x * variance_y)
    
    return correlation_coefficient