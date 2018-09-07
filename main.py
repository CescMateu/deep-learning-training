import numpy as np
from aux_functions import *
from main_functions import *

np.random.seed(1)


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

# Load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


# Run the model
AL, cache, params, grads = model(train_x, train_y, layers_dims, num_iterations = 500, learning_rate = 0.01, lambd = 0, verbose = True)
