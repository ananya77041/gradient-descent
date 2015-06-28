#!/user/bin/python

import gradient_descent as gd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log, e

# Load data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:,:2]
y = data[:,-1]
m = len(y)

ones = np.ones((m,len(X[0,:])+1))
ones[:,1:] = X
X = np.transpose(ones)

theta = np.zeros((len(X[:,0]),1))

cost, grad = gd.computeLogisticCost(X, y, theta)
print(cost)