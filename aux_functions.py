import numpy as np

# Auxiliary functions

def sigmoid(Z):

	A = 1 / (1 + np.exp(-Z))

	return(A)


def tanh(Z):

	A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

	return(A)


def relu(Z):

	A = np.maximum(Z, 0)

	return None

