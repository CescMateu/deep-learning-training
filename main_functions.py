import numpy as np
from aux_functions import *
from testCases import *

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
	X, parameters = L_model_forward_test_case_2hidden()
	AL, caches = forward_propagation_L(X, parameters, activation_hidden='relu', activation_output='sigmoid')

	'''

	L = len(parameters) // 2	# Number of layers (including input and output)
	n_x = X.shape[0]			# Number of features
	m = X.shape[1]				# Number of examples
	cache = {}					# Initialize a dictionary in which all the Z's and A's will be saved for later use in the backpropagation step
	cache['A0'] = X 			# Initialize A_prev with X for the first iteration

	# Hidden layers
	for l in range(1, L):
		cache['Z' + str(l)] = linear_forward_propagation(cache['A' + str(l-1)], parameters['W' + str(l)], parameters['b' + str(l)])
		cache['A' + str(l)] = activation_forward_propagation(cache['Z' + str(l)], activation_hidden)

	# Output layer
	cache['Z' + str(L)] = linear_forward_propagation(cache['A' + str(L-1)], parameters['W' + str(L)], parameters['b' + str(L)])
	AL = activation_forward_propagation(cache['Z' + str(L)], activation_output)

	AL = AL.reshape(1, X.shape[1])
	assert(AL.shape == (1, X.shape[1]))

	return(AL, cache)

def compute_cost(Y, AL, parameters, lambd):
	'''

	Unit tests:
	Y, AL = compute_cost_test_case()
	print("cost = " + str(compute_cost(Y, AL, parameters = [1,2], lambd = 0)))

	'''

	L = len(parameters) // 2
	m = Y.shape[1]
	
	# Basic cost function
	J = - (1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))


	# Regularization part
	W_norm = 0
	if lambd > 0:

		for l in range(1, L):
			W_l = np.linalg.norm(parameters['W' + str(l)])
			W_norm = W_norm + W_l

		reg =  (lambd / (2*m)) * W_norm
		J = J + reg

	J = np.squeeze(J)
	assert(J.shape == ())

	return(J)



def backward_propagation_layer(dA, Z, W, A_prev, activation):
	'''
	'''

	m = A_prev.shape[1] # Number of examples

	# Compute dZ
	if activation == 'sigmoid':
		dZ = sigmoid_backward(dA, Z)

	elif activation == 'tanh':
		raise ValueError('tanh_backward() has not been implemented yet')
		dZ = tanh_backward(dA, Z)

	elif activation == 'relu':
		dZ = relu_backward(dA, Z)

	# Compute dW and db
	dW = (1/m) * np.dot(dZ, A_prev.T)
	db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)

	# Compute dA_prev
	dA_prev = np.dot(W.T, dZ)
	

	return dZ, dW, db, dA_prev

def backward_propagation_L(Y, AL, parameters, cache, activation_hidden, activation_output):

	'''
	Description:

	Parameters:
	- AL: Array containing the values for the activation output from the last layer (probabilities of the clases)
	- parameters: Dictionary containing the values for W and b for each layer
	- cache: Dictionary containing the values for A and Z of the previous layers
	'''

	# Initialise some parameters
	L = len(parameters) // 2
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

	# Create a new dictionary in which to save the gradients
	grads = {}

	# Compute the gradient of the output layer as initialization
	grads['dA' + str(L)] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	for l in range(L, 0, -1):

		# Decide which activation function to use
		if l == L:
			activation_l = activation_output
		else:
			activation_l = activation_hidden

		# Backward propagation for layer l
		dZ, dW, db, dA_prev = backward_propagation_layer(
			dA = grads['dA' + str(l)], 
			Z = cache['Z' + str(l)],
			W = parameters['W' + str(l)],
			A_prev = cache['A' + str(l-1)],
			activation = activation_l
			)

		# Save the results in the grads dictionary
		grads['dZ' + str(l)] = dZ
		grads['dW' + str(l)] = dW
		grads['db' + str(l)] = db
		grads['dA' + str(l - 1)] = dA_prev

	return(grads)


def update_parameters(parameters, grads, learning_rate):
	'''
	'''	
	L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter.
	for l in range(L):

		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

	return parameters



def model(X, Y, layers_dims, num_iterations = 100, learning_rate = 0.01, lambd = 0, activation_hidden = 'relu', activation_output = 'sigmoid', verbose = False):
	'''
	'''

	# Initialize parameters
	params = initialize_parameters_L(layers_dims)

	for i in range(num_iterations):

		# Forward propagation
		AL, cache = forward_propagation_L(X, params, activation_hidden, activation_output)

		# Compute cost
		cost = compute_cost(Y, AL, params, lambd)

		# Backward propagation
		grads = backward_propagation_L(Y, AL, params, cache, activation_hidden, activation_output)

		# Update parameters with GD
		params = update_parameters(params, grads, learning_rate)

		# Print the cost every 20 iterations
		if verbose and (i % 100 == 0):
			print('Cost after iteration {}: {}'.format(i, np.squeeze(cost)))

	return(AL, cache, params, grads)



if __name__ == '__main__':
	import doctest
	doctest.testmod()