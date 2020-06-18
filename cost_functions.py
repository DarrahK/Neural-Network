# This is where I am storing all of my activivation function for my neural network

import numpy as np

# Cost functions

def cost(output, correct_data, prime = False):
    if prime:
        return np.subtract(correct_data, output)
    return np.sum(np.square(np.subtract(correct_data, output))) /2

