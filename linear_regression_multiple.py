#!/user/bin/python

# Market price of houses
# Example of linear regression with multiple variables

import gradient_descent as gd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:,:2]
y = data[:,-1]
m = len(y)

# Gradient Descent
ones = np.ones((m,len(X[0,:])+1))
ones[:,1:] = X
X = np.transpose(ones)

# Solve normal equation
theta = gd.normalEqn(np.transpose(X),y)
print('Theta found by solving normal equation: ')
print(theta)
predict = np.dot(np.array([1, 1650, 3]), theta)
print(predict)

# Gradient Descent
theta = np.zeros((len(X[:,0]),1))

iterations = 400
alpha = 0.05

X_norm, mu, sigma = gd.featureNormalize(X)

theta, J_history = gd.gradientDescent(X_norm, y, theta, alpha, iterations)
print('Theta found by gradient descent: ')
print(theta)

plt.figure(1)
plt.plot(range(len(J_history)), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

predict1 = np.dot(np.array([1, (1650-mu[1])/sigma[1], (3-mu[2])/sigma[2]]), theta)
print(predict1)