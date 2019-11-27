# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:58:30 2019

@author: mingjay
"""

import numpy as np
from numpy import *
import matplotlib.pylab as plt



def relu(x):
    return x if x >= 0 else 0
def dev_relu(x):
    return 1 if x >= 0 else 0


x = np.linspace(-10,10,100)
y1 = []
for i in range(len(x)):
    y1.append(relu(x[i]))
#check
    
print("x[49]=",x[49],"--> relu(x[49])=",relu(x[49]))
print("x[50]=",x[50],"--> relu(x[50])=",relu(x[50]))
plt.plot(x, y1, 'r', label='linspace(-10,10,10)')
plt.title('Relu Activation Function')
# create the graph
plt.show()

y2 = []
for i in range(len(x)):
    y2.append(dev_relu(x[i]))

plt.plot(x, y2, 'b', label='linspace(-10,10,10)')
plt.title('dRelu Activation Function')
# create the graph
plt.show()