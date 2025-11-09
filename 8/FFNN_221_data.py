# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:49:11 2023

@author: yunpeng
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
    
""" Distributed samples """
n_true = 80
n_false = 80
xt1 = np.random.normal(1, 0.5, n_true)
xt2 = np.random.normal(3, 0.5, n_true)
yt = np.ones((n_true, 1))
xf1 = np.random.normal(3, 0.5, n_false)
xf2 = np.random.normal(1, 0.5, n_false)
yf = np.zeros((n_false, 1))

nsp = n_true + n_false
y_data = np.vstack((yt, yf))
x1 = np.array( [np.hstack((xt1, xf1))] )
x2 = np.array( [np.hstack((xt2, xf2))] )

# Save x_sample and y_noi to a CSV file
data_to_save = np.column_stack((x1.T, x2.T, y_data))
np.savetxt('Q1_data.csv', data_to_save, delimiter=',', header='x1, x2, y_data', comments='')

# To save as a text file (space-separated values)
# np.savetxt('data.txt', data_to_save, delimiter=' ', header='x_sample y_noi', comments='')

# You can also use np.savetxt for other formats like TXT, TSV, etc.

plt.figure(1)
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y_data, cmap='b')    

