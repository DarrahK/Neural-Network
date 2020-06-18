# --- Neural Network v3.0 ---
# This is an improved Neural network package.

# Local imports 
from activation_functions import *
from cost_functions import *

# External Imports 
import numpy as np


class Network:
    """

    """

    def __init__(self, layer_s, act_func = "Sigmoid", a_const = None):                             # I could add a seed options

        self.zs          = []
        self.activations = []

        if type(layer_s) is int:
            # Allows the network to be build by modality.
            self.layers       = [layer_s]
            self.size         = 1
            self.activations  = []
            self.a_activation = []
            self.biases       = []
            self.weights      = []
        else:
            self.layers       = layer_s
            self.size         = len(layer_s)
            self.activations  = [act_func for _ in range(self.size - 1)]
            self.a_activation = [a_const for _ in range(self.size - 1)]
            self.biases       = [np.random.randn(x, 1) for x in layer_s[1:]]
            self.weights      = [np.random.randn(y, x) for x, y in zip(layer_s[:-1], layer_s[1:])]

    def add_layer(self, num_of_nodes, act_func = "Sigmoid", a_const = None):

        self.layers.append(nodes)
        self.size += 1
        self.activations.append(activation)
        self.a_activation.append(a_const)
        self.biases.append(np.random.randn(num_of_nodes, 1))
        self.weights.append(np.random.randn(num_of_nodes, self.layers[-1]))

    def change_layer(self, layer, num_of_nodes):

        self.layers[layer]  = num_of_nodes
        self.biases[layer]  = np.random.randn(num_of_nodes, 1)

        if (layer != size):
            self.weights[layer] = np.random.randn(num_of_nodes, self.layers[layer + 1])

    def feed_forward(self, input_data):                                                 # Removed output = False

        # A little check if to make sure that the the the right way
        # input_data = np.matrix(input_data).transpose()
        # Setting the input list to a numpy matrix array.

        # The code is not easy to read i.e change the input_data stuff 

        # Checking if we are able to feedforward.
        #if np.size(input_data) > self.layers[0]:
        #    return "You have too many inputs"

        self.zs          = []
        self.activations = []

        self.activations.append(input_data)

        # Feeding the data through the neural network.
        for layer in range(self.size - 1):

            # activation = W*x + b
            weight = self.weights[layer]
            bias = self.biases[layer]
            zs = np.add(np.dot(weight, input_data), bias)

            self.zs.append(zs)

            # Gathering the a data ( if needed )
            a_const = self.a_activation[layer]

            # Applying the activation function mapping
            activation_function = self.activations[layer]
            input_data = function_map(activation_function, zs, False, a_const)
            self.activations.append(input_data)

        return input_data

    def back_propagation(self, input_data, y, update = False, learning_rate = 0.05):
        
        # Maybe I should have it so my output is called gradient weights or something
        # Do I want to change that 
        # This only works with sigmoid ATM

        biases_gradient  = [np.zeros(b.shape) for b in self.biases]
        weights_gradient = [np.zeros(w.shape) for w in self.weights]

        # for the feed_forward i will need to have need have the 
        #print("input", input)


        predicted_output = self.feed_forward(input_data)
        
        # I just need to check that my cost function is correct
        # check over that delta is correct
        evaluated_cost = cost(predicted_output, y, prime = True)
        delta = np.multiply(evaluated_cost, predicted_output)


        biases_gradient[-1] = delta
        weights_gradient[-1] = np.dot(delta, self.activations[-2].transpose())

        for layer in range(2, self.size):

            z = self.zs[-layer]
            # This needs to be changed so it workes with the correct activation function
            activation_prime = sigmoid(z, True)
            delta = np.multiply(np.dot(self.weights[-layer+1].transpose(), delta), activation_prime)
            biases_gradient[-layer] = delta
            weights_copy[-layer] = np.dot(delta, self.activations[-layer-1].transpose())

        if update:
            self.weights = [w - learning_rate * w_change for w, w_change in zip(self.weights, weights_gradient)]
            self.biases  = [b - learning_rate * b_change for b, b_change in zip(self.biases, biases_gradient)]

        return (weights_gradient, biases_gradient)

    def SGD(self, data, epochs = 1, learning_rate = 0.05, training_data = None):

        for epoch in range(epochs):
            # data is a tuple,
            # I need to suffle the data
            for (input, actual_output) in data:
                self.back_propagation(input, actual_output, True, learning_rate)

            if training_data:
                print(f"Epoch: {epoch+1} / {epochs}, actuary {self.score(training_data)}")
            else:
                print(f"Epoch: {epoch+1} / {epochs}")


    def mini_batch_SGD(self, data, epochs = 1, num_batchs = 4, learning_rate  = 0.05, training_data = None):

        data_length = len(data)
        batch_size  = int(data_length / num_batchs)

        for epoch in range(epochs):
            # data is a tuple
            # I need to suffle the data
            for i in range(num_batchs):
                b_copy = [np.zeros(b.shape) for b in self.biases]
                w_copy = [np.zeros(w.shape) for w in self.weights]

                for (input, actual_output) in data[i*batch_size:(i+1)*batch_size]:
                    (nabla_w, nabla_b) = self.back_propagation(input, actual_output, learning_rate = learning_rate)
                    w_copy = [w - (learning_rate / batch_size) * w_change for w, w_change in zip(w_copy, nabla_w)]   
                    b_copy = [b - (learning_rate / batch_size) * b_change for b, b_change in zip(w_copy, nabla_b)]

            if training_data:
                print(f"Epoch: {epoch+1} / {epochs}, Batch: {i+1} / {num_batchs}, actuary {self.score(training_data)}")
            else:
                print(f"Epoch: {epoch+1} / {epochs}, Batch: {i+1} / {num_batchs}")

    def score(self, training_data):

        # that is not how you spell it so pleeseeee change it..................................................
        actuary = 0
        training_length = len(training_data)

        for (input, actual_output) in training_data:
            output  = self.feed_forward(input)
            actuary += np.dot(output.transpose(),actual_output) / self.layers[-1] 

        return actuary / training_length

    def __str__(self):
        return str(self.layers)

    def __len__(self):
        return self.size


if __name__ == "__main__":
    network = Network([3,4,2])
    #print(network.weights)
    input = np.array([[1], [1], [1]])
    output = np.array([[1], [1]])
    #print(" input: ", input)
    #output = network.feed_forward(input)
    #print(output)

    network.back_propagation(input, output, True)
    network.SGD([(input,output)], 2 , 0.05, [(input,output)])








