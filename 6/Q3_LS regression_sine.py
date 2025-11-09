# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:36:17 2022

@author: Y P Zhu

Least Squares (LS) algorithm / Maximum Likelihood (ML) algorithm
Realization of linear regression - Sine
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

# Load the data from data.csv
loaded_data = np.loadtxt('Q3_data.csv', delimiter=',', skiprows=1)  # Skip the header row

# Split the loaded data into x_sample and y_noi
x_sample = loaded_data[:, 0]
y_noi = loaded_data[:, 1]

# Find the length of the data
data_length = len(x_sample)

# Now you can use x_sample and y_noi as needed

"""
---- LS/ML regresion based on Python coding
"""
RegsMat = np.zeros((data_length, 5)) # Polynomial order is 4
for i in range(data_length):
    RegsMat[i,:] = ( [1, x_sample[i], x_sample[i]**2, 
                    x_sample[i]**3, x_sample[i]**4] )

Weights = np.linalg.inv(RegsMat.T @ RegsMat) @ RegsMat.T @ y_noi # The LS algorithm

xp_sample = np.linspace(0, 1, 1000) # Select 1000 points as the x_p
y_pred = np.zeros((1000,))
for i in range(1000):
    RegsVec = ( np.array([1, xp_sample[i], xp_sample[i]**2, 
                  xp_sample[i]**3, xp_sample[i]**4]) )
    y_pred[i] = RegsVec @ Weights

plt.figure(1)    
plt.plot(x_sample, y_noi, 'g.', label='Sampling')
plt.plot(xp_sample, y_pred, 'r--', label='Prediction')
plt.xlabel("sampling points")
plt.ylabel("Function values")
# plt.legend() # Show the labels
# plt.show() # Optional in Spyder (interactive mode)

"""
---- LS regresion based on Python package
"""
# from scipy.optimize import leastsq # Import leastsquare function

# def fit_func(weights_vab, samples): # Polynomial function
#     f = np.poly1d(weights_vab)
#     return f(samples)
 
# def residuals_func(weights_vab, y_noi, x_sample): # Residual function
#     ret = fit_func(weights_vab, x_sample) - y_noi
#     return ret
 
# np.random.seed()
# Weights_init = np.random.randn(5) # Polynomial order is 4 [x^4 ... x 1]
# Weights = leastsq(residuals_func, Weights_init, args=(y_noi, x_sample))

# xp_sample = np.linspace(0, 1, 1000) # Select 1000 points as the xp_sample
# y_pred = fit_func(Weights[0], xp_sample)

# plt.figure(2)    
# plt.plot(x_sample, y_noi, 'g.', label='Sampling')
# plt.plot(xp_sample, y_pred, 'r--', label='Prediction')
# plt.xlabel("x1")
# plt.ylabel("y")
# plt.legend() # Show the labels
# # plt.show() # Optional in Spyder (interactive mode)



