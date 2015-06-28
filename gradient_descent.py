#!/user/bin/python

import numpy as np

# computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
	m = len(y) # number of training examples
	sum = 0
	for i in range(m):
		x_i = X[:,i]
		h_theta_i = np.dot(x_i,theta)
		y_i = y[i]
		sum = sum + (h_theta_i - y_i)**2
	return sum/(2*m)

# updates theta by taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros(num_iters)

	for iter in range(num_iters):
		theta_new = np.zeros((len(theta),1))
		for j in range(len(theta)):
			sum = 0
			for i in range(m):
				x_i = X[:,i]
				h_theta_i = np.dot(x_i,theta)
				y_i = y[i]
				x_j_i = X[j,i]
				sum = sum + (h_theta_i-y_i)*x_j_i
			theta_new[j] = theta[j] - alpha*sum/m
		theta = theta_new
		J_history[iter] = computeCost(X,y,theta)

	return theta, J_history

def featureNormalize(X):
	X_norm = X;
	mu = np.zeros(len(X[:,0]))
	sigma = np.zeros(len(X[:,0]))

	for i in range(len(mu)):
		mu[i] = np.mean(X[i,:])
		if i == 0:
			sigma[i] = 1
		else:
			sigma[i] = np.std(X[i,:])
			X_norm[i,:] = (X_norm[i,:] - mu[i])/sigma[i]

	return X_norm, mu, sigma

def normalEqn(X, y):
	theta = np.zeros((len(X[:,0]),1))
	X_t = np.transpose(X)
	inv = np.linalg.inv(np.dot(X_t,X))
	theta = np.dot(np.dot(inv,X_t),y)
	return theta