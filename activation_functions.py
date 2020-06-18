# This is where I am storing all of my activivation function for my neural network

import numpy as np

# This is used to have function mapping of the activation functions.
# If you want to add another function please add it into the dictorionary inside the function_map.
 

def function_map(function_name, x, prime = False, a = None):
    functions = {
    "Sigmoid": sigmoid(x, prime),
    "Tanh": tanh(x, prime),
    "RelU": relu(x, prime),
    "ArcTan": arctan(x, prime),
    "ID": id(x, prime),
    "RPeLU": rpelu(x, a, prime),
    "ELU": elu(x, a, prime),
    "SolfPlus": solfplus(x, prime)
    }
    return functions[function_name]


 # Activation functions

def sigmoid(x, prime = False):
    if prime:
       	return np.multiply(sigmoid(x), (1-sigmoid(x)))
    return 1 / (1 + np.exp(-x))


def tanh(x, prime = False):
    if prime:
        # does this work
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
    # Does this work?
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
    # I need to find a way to make this better.
    # However this works because it is a n by 1 matrix.
    for i in range(x.shape[0]):
        if x[i] < 0:
            x[i] *= a
    return x


def elu(x, a, prime = False):
    if prime:
        x[x >= 0] = 1
        # I need to find a way to make this better.
        #x[x < 0] = ELU(x, a) + a
        for i in range(x.shape[0]):
            if x[i] < 0:
                x[i] = elu(x[i], a) + a
        return x
    # x[x < 0] = a * ( np.exp(x) - 1)
    for i in range(x.shape[0]):
        if x[i] < 0:
            x[i] = a * (np.exp(x[i]) - 1)
    return x


def solfplus(x, prime = False):
    if prime:
        return sigmoid(x)
    return np.log(1 + np.exp(x))



