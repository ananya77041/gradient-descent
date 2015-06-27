#!/user/bin/python

import numpy as np
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:,0]
y = data[:,1]
m = len(y)

# computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
	m = len(y) # number of training examples
	sum = 0
	for i in range(m):
		h_theta_i = np.dot(np.transpose(theta),X[:,i])
		y_i = y[i]
		sum = sum + (h_theta_i - y_i)**2
	return sum/(2*m)

# updates theta by taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros(num_iters)

	for iter in range(num_iters):
		theta_new = np.zeros(2)
		for j in range(len(theta)):
			sum = 0
			for i in range(m):
				h_theta_i = np.dot(np.transpose(theta),X[:,i])
				y_i = y[i]
				x_j_i = X[j,i]
				sum = sum + (h_theta_i-y_i)*x_j_i
			theta_new[j] = theta[j] - alpha*sum/m
		theta = theta_new
		J_history[iter] = computeCost(X,y,theta)

	return theta, J_history


# Plot the data
plt.figure(1)
plt.plot(X,y,'rx', ms=10)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

# Gradient Descent
X = np.array([np.ones(m), data[:,0]])
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ')
print(theta[0], theta[1])


plt.hold(True)
plt.plot(X[1,:], np.dot(np.transpose(X),theta), '-')
plt.legend(('Training Data', 'Linear Regression'), loc='lower right')


predict1 = np.dot(np.array([1, 3.5]), theta)
predict2 = np.dot(np.array([1, 7]), theta)
print("Predicted profits for 35000 people: ", predict1)
print("Predicted profits for 70000 people: ", predict2)

# Visualizing cost function J
theta0_vals = np.linspace(-10,10,num=100)
theta1_vals = np.linspace(-1,4,num=100)
surf_X, surf_Y = np.meshgrid(theta0_vals, theta1_vals)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
	for j in range(len(theta1_vals)):
		t = np.transpose(np.array([theta0_vals[i], theta1_vals[j]]))
		J_vals[i,j] = computeCost(X, y, t)

# Surface plot of J
fig = plt.figure(2)
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(surf_X, surf_Y, np.transpose(J_vals), antialiased=False, cmap='coolwarm', linewidth=0)
plt.xlabel('theta0')
plt.ylabel('theta1')

# Contour plot
fig.add_subplot(122)
plt.contour(surf_X, surf_Y, np.transpose(J_vals), np.logspace(-2, 3, num=20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.hold(True)
plt.plot(theta[0], theta[1], 'rx', ms=10, linewidth=2)
plt.show()