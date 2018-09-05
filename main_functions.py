import numpy as np
from aux_functions import *

def initialize_parameters_L(layers_dims):

	'''
	Description:
	Initialize the W and b parameters for a NN with as many hidden layers as the length of the layers_dims vector

	Parameters:
	- layers_dims: List containing the sizes of all the layers in the NN (including the input and output layers)
	
	Unit tests:
	>>> layers_dims = [5, 4, 3, 1]
	>>> param = initialize_parameters_L(layers_dims)
	>>> param['W1'].shape
	(4, 5)
	>>> np.sum(param['b2'])
	0.0
	'''

	parameters = {}
	L = len(layers_dims)

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

	return parameters


def linear_forward_propagation(A, W, b):

	'''
	Description:
	Compute the linear part of the forward propagation for one layer l with parameters W and b

	Parameters:
	- A: Output array from layer l-1
	- W: Weights W in layer l
	- b: Bias term b in layer l

	Unit tests:
	>>> A = np.array([[1, 2, 3], [4, 5, 6]])      # (2 x 3)
	>>> W = np.array([[0, 1], [4, 2], [1, -1]])   # (3 x 2)
	>>> b = np.array([1, 1, 1]).reshape(3, 1)     # (3 x 1). Note: Broadcasting
	>>> Z = linear_forward_propagation(A, W, b)
	>>> Z.shape
	(3, 3)
	'''

	assert(W.shape[1] == A.shape[0])

	Z = np.dot(W, A) + b

	return(Z)

def activation_forward_propagation(Z, act_function = 'tanh'):

	'''
	Description:
	Compute the activation part of the forward propagation for one particular layer l

	Parameters:
	- Z: Result array from the linear forward propagation from the same layer l
	- act_function: Activation function used. Options: 'sigmoid', 'tanh', 'relu'

	Unit tests:
	'''

	if act_function == 'sigmoid':
		A = sigmoid(Z)
	elif act_function == 'tanh':
		A = tanh(Z)
	elif act_function == 'relu':
		A = relu(Z)
	else:
		raise ValueError('The name of the activation function does not exist. Please try again. Options: "sigmoid", "tanh", "relu".')

	return(A)



def forward_propagation_L(X, parameters, activation_hidden, activation_output):

	'''
	Description:
	Forward propagation step across all the neurons, returning the result of the output layer

	Parameters:
	- X: Array with the examples (n_x, m)
	- parameters: Dictionary with the parameters W and b that will be used
	- activation_hidden: Activation function used in the hidden layer/s. Options: 'sigmoid', 'relu', 'tanh'
	- activation_output: Activation function used in the output layer. Options: 'sigmoid', 'relu', 'tanh'

	Unit tests:

	'''

	L = len(parameters) // 2	# Number of layers (including input and output)
	n_x = X.shape[0]			# Number of features
	m = X.shape[1]				# Number of examples
	cache = {}					# Initialize a dictionary in which all the Z's and A's will be saved for later use in the backpropagation step
	cache['A0'] = X 			# Initialize A_prev with X for the first iteration

	# Hidden layers
	for l in range(1, L):
		print('Computing hidden layer: ' + str(l))
		cache['Z' + str(l)] = linear_forward_propagation(cache['A' + str(l-1)], parameters['W' + str(l)], parameters['b' + str(l)])
		cache['A' + str(l)] = activation_forward_propagation(cache['Z' + str(l)], activation_hidden)

	# Output layer
	print('Computing output layer: ' + str(L))
	cache['Z' + str(L)] = linear_forward_propagation(cache['A' + str(L-1)], parameters['W' + str(L)], parameters['b' + str(L)])
	A_output = activation_forward_propagation(cache['Z' + str(L)], activation_output)

	return(A_output, cache)

def model(X, Y, layers_dims, activation_hidden = 'tanh', activation_output = 'sigmoid'):
	'''
	'''

	# Initialize parameters
	params = initialize_parameters_L(layers_dims)

	# Forward propagation
	A = forward_propagation_L(X, params, activation_hidden, activation_output)

	return(A)


X = np.array([[1, 1, 2], [0, 0, 1], [0, 0, 0]])
Y = np.array([0, 1, 1])
layer_dims = [X.shape[0], 2, 1]
print(model(X, Y, layer_dims))










if __name__ == '__main__':
	import doctest
	doctest.testmod()