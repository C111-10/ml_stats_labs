# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:40:50 2022

@author: Y P Zhu

Regression illustration
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

""" Few samples """
nsp=11
x1_sample = np.linspace(0, 1, nsp) # Select nsp points as the x_sample
x2_sample = np.linspace(0, 1, nsp) # Select nsp points as the x_sample
y_tag = np.zeros((nsp, nsp))
for i in range(nsp):
    for j in range(nsp):
        y_tag[j,i] = 3 * x1_sample[i]**5 + 2 * x2_sample[j]**2 + 1 # The target function

plt.figure(1) 
[X1, X2] = np.meshgrid(x1_sample, x2_sample)
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, y_tag, color='g')
plt.legend() # Show the labels
plt.show() # Optional in Spyder (interactive mode)



