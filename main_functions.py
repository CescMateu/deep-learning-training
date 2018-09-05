import numpy as np



def initialize_parameters_L(layers_dims):

	'''
	Parameters:
		- layers_dims: List containing the sizes of all the layers in the NN (including the input and output layers)
	
	Initialize the W and b parameters for a NN with as many hidden layers as the length of the layers_dims vector

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
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

	return parameters


layers_dims = [5, 4, 3, 1]
param = initialize_parameters_L(layers_dims)





if __name__ == '__main__':
	import doctest
	doctest.testmod()