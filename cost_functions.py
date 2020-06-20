# This is where I am storing all of my activivation function for my neural network

import numpy as np

# Cost functions

def cost(output, correct_data, prime = False):
    if prime:
    	return np.subtract(output, correct_data)
    return np.sum(np.square(np.subtract(output, correct_data))) / 2

