# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:05:34 2022

@author: Y P Zhu

2-class Logistic classification 
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

def sigmoid(Rmat_W):
    return 1 / (1 + np.exp(-Rmat_W))

def gradAscent(RegsMat,y_data):
    [m, n] = np.shape(RegsMat)
    alpha = 0.01
    maxCycles = 500
    weights_vab = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(RegsMat @ weights_vab)
        err = (h - y_data)
        weights_vab = weights_vab - alpha * RegsMat.T @ err
    return weights_vab

""" Logistic regression 1-D """
# n_true=20
# n_false=20
# xt = np.random.uniform(0, 1, n_true)
# yt = np.ones((n_true, 1))
# xf = np.random.uniform(2, 3, n_false)
# yf = np.zeros((n_false, 1))

# nsp = n_true + n_false
# y_data = np.vstack((yt, yf))
# x1 = np.hstack((xt, xf))

# RegsMat = np.zeros((nsp, 2)) # Polynomial order is 1
# for i in range(nsp):
#     RegsMat[i,:] = [1, x1[i]]

# Weights = gradAscent(RegsMat,y_data)

# x1_p = np.linspace(0, 3, 50)
# RegsVec_p = np.zeros((1, 2)) # Polynomial order is 1
# y_fit = np.zeros((50, 1))
# for i in range(50):
#     RegsVec_p = np.array( [1, x1_p[i]] )
#     y_fit[i] = sigmoid(RegsVec_p @ Weights)

# plt.figure(1)    
# plt.plot(xt, yt, 'g+', label='True')
# plt.plot(xf, yf, 'b.', label='False')
# plt.plot(x1_p, 0.5+np.zeros((50,)), 'b--', label='0-axis')
# plt.plot(x1_p, y_fit, 'r', label='Prediction')
# plt.xlabel("Variable values")
# plt.ylabel("class values")
# # plt.legend() # Show the labels
# # plt.show() # Optional in Spyder (interactive mode)

# """ Logistic regression 2-D """
# n_true = 80
# n_false = 80
# xt1 = np.random.normal(1, 0.5, n_true)
# xt2 = np.random.normal(3, 0.5, n_true)
# yt = np.ones((n_true, 1))
# xf1 = np.random.normal(3, 0.5, n_false)
# xf2 = np.random.normal(1, 0.5, n_false)
# yf = np.zeros((n_false, 1))

# nsp = n_true + n_false
# y_data = np.vstack((yt, yf))
# x1 = np.hstack((xt1, xf1))
# x2 = np.hstack((xt2, xf2))

# Load the data from data.csv
loaded_data = np.loadtxt('Q5_data.csv', delimiter=',', skiprows=1)  # Skip the header row

# Split the loaded data into x_sample and y_noi
x1= loaded_data[:, 0]
x2= loaded_data[:, 1]
y_data = loaded_data[:, 2]
y_data = y_data.reshape(-1, 1)

# Find the length of the data
data_length = len(x1)

# Now you can use x_sample and y_noi as needed


RegsMat = np.zeros((data_length, 3)) # Polynomial order is 1
for i in range(data_length):
    RegsMat[i,:] = [1, x1[i], x2[i]]

Weights = gradAscent(RegsMat,y_data)

x1_p = np.linspace(0, 4.5, 10)
x2_p = np.linspace(0, 4.5, 9)
RegsVec_p = np.zeros((1, 3)) # Polynomial order is 1
y_fit = np.zeros((9, 10))
for i in range(10):
    for j in range(9):
        RegsVec_p = np.array( [1, x1_p[i], x2_p[j]] )
        y_fit[j,i] = sigmoid(RegsVec_p @ Weights)

plt.figure(1) 
[X1, X2] = np.meshgrid(x1_p, x2_p)
ax = plt.axes(projection='3d')
ax.plot_wireframe(X1, X2, y_fit, color='g')
ax.scatter3D(x1, x2, y_data, cmap='b')
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("y")
# # plt.legend() # Show the labels
# # plt.show() # Optional in Spyder (interactive mode)

plt.figure(2)
x1_bound = np.linspace(0, 4.5, 1000)
x2_bound = np.zeros((1000,))
for i in range(1000):
    RegsVec_bound = np.array([1, x1_bound[i]])
    x2_bound[i] = - RegsVec_bound @ Weights[0:2] / Weights[2] # a+b*x1+c*x2=0 --> x2=-(a+b*x1)/c

plt.plot(x1[0:80], x2[0:80], 'g+', label='Class A')
plt.plot(x1[80:], x2[80:], 'y.', label='Class B')
plt.plot(x1_bound, x2_bound, 'b-')
# # plt.legend() # Show the labels
# # plt.show() # Optional in Spyder (interactive mode)