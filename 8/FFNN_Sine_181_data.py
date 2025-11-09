# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:57:41 2023

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

""" Distributed samples """
nsp=10

x_sample = np.linspace(0, 1, nsp) # Select nsp points as the x
y_data = np.sin(2 * np.pi * x_sample) # The target function

# Save x_sample and y_noi to a CSV file
data_to_save = np.column_stack((x_sample, y_data))
np.savetxt('Q2_data.csv', data_to_save, delimiter=',', header='x_sample, y_data', comments='')

# To save as a text file (space-separated values)
# np.savetxt('data.txt', data_to_save, delimiter=' ', header='x_sample y_noi', comments='')

# You can also use np.savetxt for other formats like TXT, TSV, etc.

plt.figure(1)
plt.plot(x_sample, y_data, 'b.')
plt.show()
