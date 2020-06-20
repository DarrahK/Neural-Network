# --- Neural Network v3.0 ---
# This is an improved Neural network package.

# Local imports 
from activation_functions import *
from cost_functions import *
from save_and_load import *
from preparing_data import *

# External Imports 
import numpy as np
import random

class Network:

    def __init__(self, layer_s, act_func = "Sigmoid", a_const = None): 

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

        self.layers.append(num_of_nodes)
        self.size += 1
        self.activations.append(act_func)
        self.a_activation.append(a_const)
        self.biases.append(np.random.randn(num_of_nodes, 1))
        self.weights.append(np.random.randn(num_of_nodes, self.layers[-2]))

    def change_layer(self, layer, num_of_nodes, act_func = None, a_const = None):

        if act_func:
            self.activations[layer] = act_func

        if a_const:
            self.a_activation[layer] = a_const

        self.layers[layer]  = num_of_nodes
        # if the layer we want to change is 0 we do not need to rechange the biases for that layer
        if layer != 0:
            self.biases[layer - 1 ]  = np.random.randn(num_of_nodes, 1)

        # There is 3 cases where we need to change the weigths
        if layer == 0:
            self.weights[layer] = np.random.randn(self.layers[1], num_of_nodes)
        elif layer == self.size - 1:
            self.weights[layer - 1] = np.random.randn(num_of_nodes, self.layers[-2])
        else:
            self.weights[layer - 1] = np.random.randn(num_of_nodes, self.layers[layer - 1])
            self.weights[layer]     = np.random.randn(self.layers[layer + 1], num_of_nodes)


    def feed_forward(self, input_data):                      

        self.zs          = []
        self.activations = []

        self.activations.append(input_data)

        for layer in range(self.size - 1):

            # activation = W*x + b
            weight = self.weights[layer]
            bias = self.biases[layer]
            zs = np.add(np.dot(weight, input_data), bias)

            self.zs.append(zs)

            a_const = self.a_activation[layer]
            activation_function = self.activations[layer]

            input_data = function_map(activation_function, zs, False, a_const)
            self.activations.append(input_data)

        return input_data

    def back_propagation(self, input_data, y, update = False, learning_rate = 0.05):
        
        biases_gradient  = [np.zeros(b.shape) for b in self.biases]
        weights_gradient = [np.zeros(w.shape) for w in self.weights]

        predicted_output = self.feed_forward(input_data)
        
        evaluated_cost = cost(predicted_output, y, prime = True)
        # TODO: I need to make sure this works with other functions
        delta = np.multiply(evaluated_cost, sigmoid(self.zs[-1], True))

        biases_gradient[-1]  = delta
        weights_gradient[-1] = np.dot(delta, self.activations[-2].transpose())

        for layer in range(2, self.size):

            z = self.zs[-layer]
            a_const = self.a_activation[-layer]
            activation_function = self.activations[-layer]
            activation_prime = function_map(activation_function, z, True, a_const)

            delta = np.multiply(np.dot(self.weights[-layer+1].transpose(), delta), activation_prime)
            biases_gradient[-layer] = delta
            weights_gradient[-layer] = np.dot(delta, self.activations[-layer-1].transpose())

        if update:
            self.weights = [w - learning_rate * w_change for w, w_change in zip(self.weights, weights_gradient)]
            self.biases  = [b - learning_rate * b_change for b, b_change in zip(self.biases, biases_gradient)]

        return (weights_gradient, biases_gradient)

    def SGD(self, input_data, epochs = 1, learning_rate = 0.05, training_data = None):

        for epoch in range(epochs):
            random.shuffle(input_data)

            for (input, actual_output) in input_data:

                self.back_propagation(input, actual_output, True, learning_rate)

            if training_data:
                print(f"Epoch: {epoch + 1} / {epochs}, Average Costs {self.score(training_data)}")

            else:
                print(f"Epoch: {epoch + 1} / {epochs}")


    def mini_batch_SGD(self, input_data, epochs = 1, num_batchs = 4, learning_rate  = 0.05, training_data = None):

        data_length = len(input_data)
        batch_size  = int(data_length / num_batchs)

        for epoch in range(epochs):

            random.shuffle(input_data)
            for i in range(num_batchs):
                b_copy = [np.zeros(b.shape) for b in self.biases]
                w_copy = [np.zeros(w.shape) for w in self.weights]

                for (input, actual_output) in input_data[i*batch_size:(i + 1)*batch_size]:
                    (nabla_w, nabla_b) = self.back_propagation(input, actual_output, learning_rate = learning_rate)
                    w_copy = [w - (learning_rate / batch_size) * w_change for w, w_change in zip(w_copy, nabla_w)]   
                    b_copy = [b - (learning_rate / batch_size) * b_change for b, b_change in zip(w_copy, nabla_b)]

            if training_data:
                print(f"Epoch: {epoch + 1} / {epochs}, Batch: {i + 1} / {num_batchs}, Average Cost: {self.score(training_data)}")
            else:
                print(f"Epoch: {epoch + 1} / {epochs}, Batch: {i + 1} / {num_batchs}")

    def score(self, training_data):

        costs = 0
        training_length = len(training_data)

        for (input, actual_output) in training_data:
            output  = self.feed_forward(input)
            costs += cost(output, actual_output)

        return costs / training_length

    def __str__(self):
        return str(self.layers)

    def __len__(self):
        return self.size
