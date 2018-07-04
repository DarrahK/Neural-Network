
# Standard lib
import random
import csv
import copy

# External lib
import numpy as np


class Network:

    def __init__(self, sizes):
        """
        This function creates the neural network. Sizes is a list that contains the input of how many nodes there is on
        each layer of the network.
        """
        self.sizes = sizes
        self.layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]


    def feed_forward(self, a_data):
        """
        This feeds the A_data through the neural network.
        """
        # This converts it to an np array so we are able to use np matrix mul
        self.a_data = []
        self.nodes = []
        self.zs = []
        self.a_data = np.matrix(a_data).transpose()
        self.nodes.append(self.a_data)
        for x in range(self.layers - 1):
            self.zs.append(np.add(np.dot(self.weights[x], self.nodes[x]), self.biases[x]))
            self.nodes.append(sigmoid(self.zs[x]))

    def back_prop(self, x, y):
        self.feed_forward(x)
        y_tran = np.matrix(y).transpose()
        delta = np.multiply(cost(self, y_tran), sigmoid(self.zs[-1],True), True)
        biases_copy = copy.deepcopy(self.biases)
        weights_copy = copy.deepcopy(self.weights)
        biases_copy[-1] = delta
        nodes_2 = np.matrix(self.nodes[-2])
        weights_copy[-1] = np.dot(delta, nodes_2.transpose())
        for back in range(1, self.layers - 1):
            weight = np.matrix(self.weights[-back])
            delta = np.multiply(np.dot(weight, delta), sigmoid(self.zs[-back-1]), True)
            biases_copy[-back] = delta
            nodes = np.matrix(self.nodes[-back-1])
            weights_copy[-back] = np.dot(delta, nodes.transpose())

        return biases_copy, weights_copy


# Other functions that are needed.
def sigmoid(x, prime=False):
    """
    Sigmoid function for a matrix applied to every element.
    """
    if prime:
        return np.dot(sigmoid(x),(1-sigmoid(x,)))
    x = np.matrix(x)
    return 1 / (1 + np.exp(-x))

def cost(network, correct_data,prime = False):
    """
    Works for out the cost function
    """
    if prime:
        return np.subtract(network.nodes[-1], correct_data)
    return np.sum(np.square(np.subtract(network.nodes[network.layers - 1], correct_data)))




