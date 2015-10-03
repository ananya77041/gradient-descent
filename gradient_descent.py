#!/user/bin/python

import numpy as np


def sigmoid(z):
	'''
	Calculates the sigmoid function
		INPUT: float
		OUTPUT: float
	'''
	return 1. / (1 + e ** (-z))


def computeCost(X, y, theta):
	'''
	Computes cost of using theta as the parameter for LinReg on X and y
		INPUT: ndarray, ndarray, ndarray
		OUTPUT: float
	'''
	return sum([(np.dot(X[:, i], theta) - y[i]) ** 2
				for i in xrange(len(y))]) / (2 * len(y))


def computeLogisticCost(X, y, theta):
	'''
	Computes logistic cost of using theta as the parameter for LinReg on X and y
		INPUT: ndarray, ndarray, ndarray
		OUTPUT: float
	'''
	grad = np.zeros(len(theta))

	for j in xrange(len(theta)):
		sum_grad = 0
		sum = 0
		for i in range(len(y)):
			h_theta_i = sigmoid(np.dot(X[:, i], theta))
			sum_grad += (h_theta_i - y[i]) * X[j, i]
			sum += (- y[i] * np.log(h_theta_i) - (1 - y[i]) * np.log(1 - h_theta_i))

		cost = sum / float(len(y))		
		grad[j] = sum_grad / float(len(y))

	return cost, grad


def gradientDescent(X, y, theta, alpha, num_iters):
	'''
	Updates theta by taking num_iters gradient descent steps with learning
	rate alpha.
		INPUT: ndarray, ndarray, ndarray, float, int
		OUTPUT: ndarray, ndarray
	'''
	J_history = np.zeros(num_iters)

	for iter in xrange(num_iters):
		theta_new = np.zeros((len(theta), 1))

		for j in xrange(len(theta)):
			sum = 0

			for i in xrange(len(y)):
				h_theta_i = np.dot(X[:, i], theta)
				sum += (h_theta_i - y[i]) * X[j, i]

			theta_new[j] = theta[j] - alpha * sum / float(len(y))

		theta = theta_new
		J_history[iter] = computeCost(X, y, theta)

	return theta, J_history


def featureNormalize(X):
	'''
	Normalizes features in X based on mean and std deviation
		INPUT: ndarray
		OUTPUT: ndarray, float, float
	'''
	X_norm = X
	mu = np.zeros(len(X[:, 0]))
	sigma = np.zeros(len(X[:, 0]))

	for i in xrange(len(mu)):
		mu[i] = np.mean(X[i, :])
		if i == 0:
			sigma[i] = 1
		else:
			sigma[i] = np.std(X[i, :])
			X_norm[i, :] -= mu[i] / sigma[i]

	return X_norm, mu, sigma


def normalEqn(X, y):
	'''
	Performs linear regression via normal equation
		INPUT: ndarray, ndarray
		OUTPUT: ndarray
	'''
	return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
