# This is where I am storing all of my activivation function for my neural network

import numpy as np

# This is used to have function mapping of the activation functions.
# If you want to add another function please add it into the dictorionary inside the function_map.
 

def function_map(function_name, x, prime = False, a = 1.0):
    functions = {
    "Sigmoid": sigmoid(x, prime),
    "RPeLU": rpelu(x, a, prime),   
    "Tanh": tanh(x, prime),
    "RelU": relu(x, prime),
    "ArcTan": arctan(x, prime),
    "ID": id(x, prime),
    "ELU": elu(x, a, prime),
    "SolfPlus": solfplus(x, prime)
    }
    return functions[function_name]
    
 # Activation functions

def sigmoid(x, prime = False):
    if prime:
       	return np.multiply(sigmoid(x), (1 - sigmoid(x)))
    return 1 / (1 + np.exp(-x))

def tanh(x, prime = False):
    if prime:
        return 1 - np.square(np.tanh(x))
    return np.tanh(x)

def relu(x, prime = False):
    if prime:
        x[x >= 0] = 1
        x[x < 0] = 0
        return x
    x[x < 0] = 0
    return x

def id(x, prime = False):
    if prime:
        x = np.ones((x.shape[0],x.shape[1]))
        return x
    return x

def arctan(x, prime = False):
    if prime:
        return 1 / (np.square(x) + 1)
    return np.arctan(x)

def rpelu(x, a, prime = False):
    if prime:
        x[x >= 0] = 1
        x[x < 0] = a
        return x
    for i in range(x.shape[0]):
        if x[i][0] < 0:
            x[i][0] = np.multiply(np.arange(a, dtype=np.float)[0], x[i][0])
    return x

def elu(x, a, prime = False):
    if prime:
        x[x >= 0] = 1

        for i in range(x.shape[0]):
            if x[i][0] < 0:
                x[i][0] = elu(x[i][0], np.arange(a, dtype=np.float)[0]) + np.arange(a, dtype=np.float)[0]
        return x

    for i in range(x.shape[0]):
        if x[i][0] < 0:
            x[i][0] = np.arange(a, dtype=np.float)[0] * (np.exp(x[i][0]) - 1)
    return x

def solfplus(x, prime = False):
    if prime:
        return sigmoid(x)
    return np.log(1 + np.exp(x))



