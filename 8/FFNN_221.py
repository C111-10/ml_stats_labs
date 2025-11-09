# -*- coding: utf-8 -*-
#"""
#Created on Sun Oct 15 14:05:27 2023
#
#@author: yunpeng
#"""
#import os

#os.system('cls')  # On Windows System

#from IPython import get_ipython
#get_ipython().run_line_magic('reset','-sf')

#"""
#Start coding
#"""
import numpy as np # Import numpy
import matplotlib.pyplot as plt # Import pyplot
# plt.close('all')

print("NN regresion based on Python coding ")

# Random data
def random_number(a,b):
    return (b-a)*np.random.normal()+a
 
# Generate m*n matrix, initial values are zeros
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill]*n)
    return np.array(a)
 
# sigmoid() function (activation function)
def sigmoid(x):
    fAct = 1/(1+np.exp(-1*x))
    return fAct

# 3-layer BP Neural Networks
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # Nodes of input, hidden, output layers
        self.num_in = num_in + 1  # Add one bias node
        self.num_hidden = num_hidden + 1   # Add one bias node
        self.num_out = num_out
        
        # Activate all nodes (vector)
        self.active_in = np.array([1.0]*self.num_in)
        self.active_hidden = np.array([1.0]*self.num_hidden)
        self.active_out = np.array([1.0]*self.num_out)
        
        # Create weight matrices
        self.wight_in = makematrix(self.num_in, self.num_hidden-1)
        self.wight_out = makematrix(self.num_hidden, self.num_out)
        
        # Weights
        W_in=[[-0.05, -0.05],
         [-2.74, -2.74],
         [2.70, 2.70]]
        W_out=[[-5.72],
         [6.06],
         [6.06]]

        for i in range(self.num_in):
            for j in range(self.num_hidden-1):
                self.wight_in[i][j] = W_in[i][j] # random_number(0.1, 0.1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = W_out[i][j] #random_number(0.1, 0.1)
              
    # Feed-forward
    def Feedforward(self, inputs):
        if np.shape(inputs)[1] != self.num_in-1:
            raise ValueError('Incorrect input numbers')
        # Input layer values
        self.active_in[1:self.num_in]=inputs
        
        # Hidden layer values
        self.sum_hidden=np.dot(self.wight_in.T,np.array([self.active_in]).T)
        self.active_hidden = np.vstack( (1, sigmoid(self.sum_hidden)) )   # Activation function
            
        # Output layer values
        self.sum_out=np.dot(self.wight_out.T,self.active_hidden)
        self.active_out = sigmoid(self.sum_out)
        return self.active_out
    
""" Distributed samples """
# Load the data from data.csv
loaded_data = np.loadtxt('Q1_data.csv', delimiter=',', skiprows=1)  # Skip the header row

# Split the loaded data into x_sample and y_noi
x1 = loaded_data[:, 0]
x1 = x1.reshape(-1, 1)
x2 = loaded_data[:, 1]
x2 = x2.reshape(-1, 1)
y_data = loaded_data[:, 2]
y_data = y_data.reshape(-1, 1)

# Find the length of the data
data_length = len(x1)

# Now you can use x_sample and y_noi as needed

pattern = np.hstack( (x1,x2) )

n = BPNN(2, 2, 1)

y_out = []
for j in pattern:
  inputs = np.array([j[0:2]])
  y_out.append( n.Feedforward(inputs) )
  
plt.figure(1)
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y_data, cmap='b')  
ax.scatter3D(x1, x2, y_out, cmap='b')  

