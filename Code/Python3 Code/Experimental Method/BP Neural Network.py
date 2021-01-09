import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)


class BP_Neural_Network:
    
    # back propagation nerual network algorithm
    
    def __init__(self):
        
        # define the number of input/ hidden/ output neurons        
        self.input_number = 0
        self.hidden_number = 0
        self.output_number = 0
        
        # define input/ hidden/ output data        
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        
        # define the initial input/ hidden weights
        self.input_weights = []
        self.hidden_weights = []
        
        # define the modified input/ hidden weights
        self.input_modify = []
        self.hidden_modify = []
        
                
    def sigmoid(self, x):
        
        # sigmoid function
        # 1 / (1 + e^(-x))
        return 1.0 / (1.0 + math.exp(-x))


    def sigmoid_derivative(self, x):
        
        # sigmoid derivative function
        # sigmoid * (1 - sigmoid)
        return x * (1 - x)


    def rand_interval(self, start, end):
        
        # define random number
        return (end - start) * random.random() + start


    def weights_matrix(self, neuron_input, neuron_output, initial_fill=0.0):
        
        # define weights matrix
        create_matrix = []
        
        for i in range(neuron_input):
            create_matrix.append([initial_fill] * neuron_output)
            
        return create_matrix

    
    def initial_setup(self, input_number, hidden_number, output_number):
        
        # initial the number of input/ hidden/ output neurons
        self.input_number = input_number + 1
        self.hidden_number = hidden_number
        self.output_number =  output_number
        
        # initial input/ hidden/ output cells
        self.input_cells = [1.0] * self.input_number
        self.hidden_cells = [1.0] * self.hidden_number
        self.output_cells = [1.0] * self.output_number
        
        # random initial input weights
        self.input_weights = self.weights_matrix(self.input_number, self.hidden_number)
        
        for i in range(self.input_number):            
            for j in range(self.hidden_number):
                self.input_weights[i][j] = self.rand_interval(-0.2, 0.2)
                
        # random initial hidden weights 
        self.hidden_weights = self.weights_matrix(self.hidden_number, self.output_number)
        
        for j in range(self.hidden_number):            
            for k in range(self.output_number):
                self.hidden_weights[j][k] = self.rand_interval(-0.2, 0.2)
        
        # initial modified input/ hidden weights
        self.input_modify = self.weights_matrix(self.input_number, self.hidden_number)
        self.hidden_modify = self.weights_matrix(self.hidden_number, self.output_number)
        

    def predict(self, input_data):
        
        # activate the input layer
        for i in range(self.input_number - 1):
            self.input_cells[i] = input_data[i]
            
        # activate the hidden layer    
        for j in range(self.hidden_number):
            sum_hidden = 0.0
            
            for i in range(self.input_number):
                sum_hidden += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = self.sigmoid(sum_hidden)
            
        # activate the output layer
        for k in range(self.output_number):
            sum_output = 0.0
            
            for j in range(self.hidden_number):
                sum_output += self.hidden_cells[j] * self.hidden_weights[j][k]
            self.output_cells[k] = self.sigmoid(sum_output)
            
        return self.output_cells[:]
    

    def back_propagate(self, learning_sample, sample_label, learning_rate, momentum):
        
        # feed forward neural network
        self.predict(learning_sample)
        
        # define output layer error
        output_deltas = [0.0] * self.output_number
        
        for k in range(self.output_number):
            output_error = sample_label[k] - self.output_cells[k]
            output_deltas[k] = self.sigmoid_derivative(self.output_cells[k]) * output_error
            
        # define hidden layer error   
        hidden_deltas = [0.0] * self.hidden_number
        
        for j in range(self.hidden_number):
            output_error = 0.0
            
            for k in range(self.output_number):
                output_error += output_deltas[k] * self.hidden_weights[j][k]
            hidden_deltas[j] = self.sigmoid_derivative(self.hidden_cells[j]) * output_error
            
        # define modified hidden weights    
        for j in range(self.hidden_number):
            for k in range(self.output_number):
                output_change = output_deltas[k] * self.hidden_cells[j]
                self.hidden_weights[j][k] += learning_rate * output_change + momentum * self.hidden_modify[j][k]
                self.hidden_modify[j][k] = output_change
                
        # define modified input weights   
        for i in range(self.input_number):
            for j in range(self.hidden_number):
                hidden_change = hidden_deltas[j] * self.input_cells[i]
                self.input_weights[i][j] += learning_rate * hidden_change + momentum * self.input_modify[i][j]
                self.input_modify[i][j] = hidden_change
                
        # define global error
        global_error = 0.0
        for k in range(len(sample_label)):
            global_error += 0.5 * (sample_label[k] - self.output_cells[k]) ** 2
            
        return global_error
    

    def training_set(self, learning_samples, sample_labels, maximum, learning_rate, momentum, error_convergence):
        
        # iterative cumulative error
        for j in range(maximum):
            global_error = 0.0  
            
            # calculate the global error of the training set
            for i in range(len(learning_samples)):
                sample_label = sample_labels[i]
                learning_sample = learning_samples[i]
                global_error += self.back_propagate(learning_sample, sample_label, learning_rate, momentum) 
            
            # global error convergence judgment
            if abs(global_error) <= error_convergence:
                break
        

    def test_set(self, learning_samples, sample_labels, input_number, hidden_number, output_number, maximum, learning_rate, momentum, error_convergence):

        # call initial_setup and training_set function
        self.initial_setup(input_number, hidden_number, output_number)
        self.training_set(learning_samples, sample_labels, maximum, learning_rate, momentum, error_convergence)
        
        # use learning samples for testing
        for learning_sample in learning_samples:
            self.predict(learning_sample)