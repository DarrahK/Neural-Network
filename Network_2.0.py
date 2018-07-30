# --- Neural Network v2.0 ---
# This is an improved Neural network package.

import numpy as np


class Network:
    """

    """

    def __init__(self, number_of_inputs, act_funct="Sigmoid"):
        if type(number_of_inputs) is int:
            # we can create a network slowly by modality.
            self.biases = []
            self.weights = []
            self.activations = []
            self.layers = [number_of_inputs]
            self.a_activation = []
        # check to see if the default if a sigmoid if not........

        # Use a different sigmoid function........
        # We can do this by cooparing one thing

    def add_layer(self, nodes, activation="Sigmoid", a=0):
        self.a_activation.append(a)
        self.activations.append(activation)
        self.biases.append(np.random.randn(nodes, 1))
        self.weights.append(np.random.randn(nodes, self.layers[-1]))
        self.layers.append(nodes)

    def change_input_layers(self, numbers_of_inputs):
        self.layers[0] = numbers_of_inputs

    def feed_forward(self, input):
        # Clearing all the data so it can be used for feeding forward.
        self.a_data = []
        self.nodes = []
        # We need the zs to be stored so we can use it in back propagation.
        self.zs = []
        # Setting the input list to a numpy matrix array.
        input_transpose = np.matrix(input).transpose()
        # Feeding the transposed input in the nodes list.
        self.nodes.append(input_transpose)
        # Feeding the data through the neural network.
        for layer in range(len(self.layers) - 1):
            # defining the weight and node for that layer.
            weight = self.weights[layer]
            node = self.nodes[layer]
            w_dot_node = np.dot(weight, node)
            zs = np.add(w_dot_node, self.biases[layer])
            # Saving the zs data in self.zs.
            self.zs.append(zs)
            # Gathering the a data( if needed )
            a = self.a_activation[layer]
            # Applying the activation function mapping
            act_funct = self.activations[layer]
            new_node = function_map(act_funct, node, a)
            # Saving the new node in the self.nodes.
            self.nodes.append(new_node)
            # FIX NODEEEEE THINGGG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __str__(self):
        return str(self.layers)

    def __len__(self):
        return len(self.layers)

# Functions that will be used as the activations.

def Sigmoid(x, prime=False):
    if prime:
        return np.multiply(Sigmoid(x), (1-Sigmoid(x)))
    return 1 / (1 + np.exp(-x))


def Tanh(x, prime=False):
    if prime:
        # does this work
        return 1 - np.square(np.tanh(x))
    return np.tanh(x)


def RelU(x, prime=False):
    if prime:
        x[x >= 0] = 1
        x[x < 0] = 0
        return x
    x[x < 0] = 0
    return x


def ID(x, prime=False):
    if prime:
        x[x] = 1
        return x
    return x


def ArcTan(x, prime=False):
    if prime:
        return 1 / (np.square(x) + 1)
    return np.arctan(x)


def RPeLU(x, a, prime=False):
    if prime:
        x[x >= 0] = 1
        x[x < 0] = a
        return x
    x[x < 0] = a * x
    return x


def ELU(x, a, prime=False):
    if prime:
        x[x >= 0] = 1
        x[x < 0] = ELU(x, a) + a
        return x
    x[x < 0] = a * ( np.exp(x) - 1)
    return x


def SolfPlus(x, prime=False):
    if prime:
        return Sigmoid(x)
    return np.log(1 + np.exp(x))


# This is used to have function mapping

def function_map(function, x, a=0):
    functions = {
    "Sigmoid": Sigmoid(x),
    "Tanh": Tanh(x),
    "RelU": RelU(x),
    "ArcTan": ArcTan(x),
    "ID":ID(x),
    "RPeLU":RPeLU(x, a),
    "ELU":ELU(x, a),
    "SolfPlus":SolfPlus(x)
    }
    return functions[function]

if __name__ == "__main__":
    A = Network(3)
    A.add_layer(3)
    A.add_layer(2)
    print(A)
