# --- Neural Network v2.0 ---
# This is an improved Neural network package.

import numpy as np


class Network:
    """

    """

    def __init__(self, layer_s, act_funct="Sigmoid", a=None):
        if type(layer_s) is int:
            # we can create a network slowly by modality.
            self.biases = []
            self.weights = []
            self.activations = []
            self.layers = [layer_s]
            self.a_activation = []

        # We can make a network faster if they don't want control.
        else:
            self.layers = layer_s
            self.size = len(layer_s)
            self.activations = list_of_same(act_funct, self.size - 1)
            self.biases = [np.random.randn(x, 1) for x in layer_s[1:]]
            self.weights = [np.random.randn(x, y) for x, y in zip(layer_s[1:], layer_s[:-1])]
            self.a_activation = list_of_same(a, self.size - 1)

    def add_layer(self, nodes, activation="Sigmoid", a=None):
        self.a_activation.append(a)
        self.activations.append(activation)
        self.biases.append(np.random.randn(nodes, 1))
        self.weights.append(np.random.randn(nodes, self.layers[-1]))
        self.layers.append(nodes)

    def change_layer(self, layer, numbers_of_nodes):
        # This changes the numbers of nodes of a layer starting at 0.
        self.layers[layer] = numbers_of_nodes

    def feed_forward(self, input_data, output=False):
        # Clearing all the data so it can be used for feeding forward.
        self.a_data = []
        self.nodes = []
        # We need the zs to be stored so we can use it in back propagation.
        self.zs = []
        # Setting the input list to a numpy matrix array.
        input_transpose = np.matrix(input_data).transpose()
        # Checking if we are able to feedforard.
        if np.size(input_transpose) > self.layers[0]:
            return "You have too many inputs"
        # Feeding the transposed input in the nodes list.
        self.nodes.append(input_transpose)
        # Feeding the data through the neural network.
        for layer in range(len(self.layers) - 1):
            # defining the weight and node for that layer.
            weight = self.weights[layer]
            node = self.nodes[layer]
            biase = self.biases[layer]
            zs = np.add(np.dot(weight, node), biase)
            # Saving the zs data in self.zs.
            self.zs.append(zs)
            # Gathering the a data ( if needed )
            a = self.a_activation[layer]
            # Applying the activation function mapping
            act_funct = self.activations[layer]
            new_node = function_map(act_funct, zs, a)
            self.nodes.append(new_node)
        # If the users wants an output.
        if output:
            # Outputs the last layer of the network if set to True.
            return new_node

    def back_prop(self, x, y, update=False):
        # Feeding the data forward so we can calculate back propagation using the correct values.
        self.feed_forward(x)
        # Working out the first propagation.
        y_tran = np.matrix(y).transpose()
        act_funct = self.activations[-1]
        a_data = self.a_activation[-1]
        delta = np.multiply(cost(self.nodes[-1], y_tran, True), function_map(act_funct, self.zs[-1], a_data, True))
        # Makes a copy of the networks biases and weights.
        biases_copy = [np.zeros(b.shape) for b in self.biases]
        weights_copy = [np.zeros(w.shape) for w in self.weights]
        biases_copy[-1] = delta
        # Making sure that the data is a NumPy matrix
        nodes_2 = np.matrix(self.nodes[-2])
        weights_copy[-1] = np.dot(delta, nodes_2.transpose())
        # Applying back propagation to the rest of the nodes.
        for back in range(1, len(self.layers) - 1):
            weight = np.matrix(self.weights[-back])
            act_funct = self.activations[-back-1]
            a_data = self.a_activation[-back-1]
            delta = np.multiply(np.dot(weight.transpose(), delta), function_map(act_funct,self.zs[-back-1], a_data, True))
            biases_copy[-back-1] = delta
            nodes = np.matrix(self.nodes[-back-2])
            weights_copy[-back-1] = np.dot(delta, nodes.transpose())
        # This updated the self.biases and self.weights to the new corrections.
        if update:
            self.biases = np.add(self.biases, biases_copy)
            self.weights = np.add(self.biases, weights_copy)
        else:
            return biases_copy, weights_copy

    def __str__(self):
        return str(self.layers)

    def __len__(self):
        return len(self.layers)

# Functions that will be used as the activations.
# Indexing might be a way that I can make it better.

def sigmoid(x, prime=False):
    if prime:
        return np.multiply(sigmoid(x), (1-sigmoid(x)))
    return 1 / (1 + np.exp(-x))


def tanh(x, prime=False):
    if prime:
        # does this work
        return 1 - np.square(np.tanh(x))
    return np.tanh(x)


def relu(x, prime=False):
    if prime:
        x[x >= 0] = 1
        x[x < 0] = 0
        return x
    x[x < 0] = 0
    return x


def id(x, prime=False):
    # Does this work?
    if prime:
        x = np.ones((x.shape[0],x.shape[1]))
        return x
    return x


def arctan(x, prime=False):
    if prime:
        return 1 / (np.square(x) + 1)
    return np.arctan(x)


def rpelu(x, a, prime=False):
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


def elu(x, a, prime=False):
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


def solfplus(x, prime=False):
    if prime:
        return sigmoid(x)
    return np.log(1 + np.exp(x))

# Cost analysis.


def cost(output, correct_data, prime=False):
    if prime:
        return 2 * np.subtract(correct_data, output)
    return np.sum(np.square(np.subtract(correct_data, output)))


# This is used to have function mapping of the activation functions.

def function_map(function_name, x, prime=False, a=None):
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


def list_of_same(item, number):
    return [item for item in range(number)]
