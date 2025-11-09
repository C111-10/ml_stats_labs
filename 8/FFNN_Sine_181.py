# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:43:49 2022

@author: Y P Zhu

NN regression algorithm - Line 
"""
import os
os.system('cls')  # On Windows System

from IPython import get_ipython
get_ipython().run_line_magic('reset','-sf')

"""
Start coding
"""
import numpy as np # Import numpy
import matplotlib.pyplot as plt # Import pyplot
# plt.close('all')

"""
---- NN regresion based on Python coding
"""

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
    fAct = (np.exp(1*x)-np.exp(-1*x))/(np.exp(1*x)+np.exp(-1*x))
    return fAct

# Derivation of the activation fucntion
def derived_sigmoid(x):
    fAct = (np.exp(1*x)-np.exp(-1*x))/(np.exp(1*x)+np.exp(-1*x))
    DfAct = (1-fAct**2)
    return DfAct

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
        
        # weights
        W_in=[[5.35, 3.12, -2.58, 0.26, -1.92, -0.79,
           2.56, -4.73],
         [-4.79, -0.41, -2.71, -0.61, -0.39, -2.85,
          -5.36, -1.47]]
        W_out=[[2.26], [-2.42], [-3.18], [-1.12],
         [0.08], [2.72], [-4.98], [1.44], [-0.06]]
        
        for i in range(self.num_in):
            for j in range(self.num_hidden-1):
                self.wight_in[i][j] = W_in[i][j] # random_number(-1, 1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = W_out[i][j] # random_number(-1, 1)
              
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
        self.active_out = self.sum_out #sigmoid(self.sum_out)

        return self.active_out

""" Distributed samples """
# Load the data from data.csv
loaded_data = np.loadtxt('Q2_data.csv', delimiter=',', skiprows=1)  # Skip the header row

# Split the loaded data into x_sample and y_data
x_sample = loaded_data[:, 0]
y_data = loaded_data[:, 1]

# Find the length of the data
data_length = len(x_sample)
# Now you can use x_sample and y_data as needed

n = BPNN(1, 8, 1) #Create neural networK 
    
xp_sample = np.linspace(0, 1, 100) # Select 100 points as the x_p
pattern = np.vstack( (xp_sample) )

y_out = []
for j in pattern:
  inputs = np.array([j[0:1]])
  y_out.append( n.Feedforward(inputs) )

yo = [item[0] for item in y_out]
# Convert yo to a numpy array if needed
yo = np.array(yo)

plt.figure(1)
plt.plot(x_sample, y_data, 'b.')
plt.plot(xp_sample,yo,'g')
plt.show()
