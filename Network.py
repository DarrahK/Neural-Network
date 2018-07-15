# --- Neural Network ---


# External lib
import numpy as np


class Network:

    def __init__(self, sizes):
        """ inits the Network class by producing random biases and weights of the sizes of each layer of the network
        :param sizes:  The input value as a list for how many nodes should be on each layer of the network.
        """
        self.sizes = sizes
        self.layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

    def feed_forward(self, a_data):
        """ This method works out the values of the nodes for the network.
        :param a_data: The input values as a list of the network.
        """
        # Clearing all the data so it can be used for feeding forward.
        self.a_data = []
        self.nodes = []
        self.zs = []
        self.a_data = np.matrix(a_data).transpose()
        self.nodes.append(self.a_data)
        # Sorts of thee zs data and then applies the sigmoid function to it and stores it into the nodes attribute.
        # This is useful for back propagation so we don't need to recalculate it later on.
        for x in range(self.layers - 1):
            self.zs.append(np.add(np.dot(self.weights[x], self.nodes[x]), self.biases[x]))
            self.nodes.append(sigmoid(self.zs[x]))

    def back_prop(self, x, y):
        """ This method applies back propagation to the network in order for it to learn.
        :param x: The input values as a list for the network.
        :param y: The output value as a list of the correct value for the input.
        :return: A matrix list of the changes to the biases and weights to get a more accurate reading.
        """
        # Feeding the data forward so we can calculate back propagation using the correct values.
        self.feed_forward(x)
        # Working out the first propagation.
        y_tran = np.matrix(y).transpose()
        delta = np.multiply(cost(self, y_tran, True), sigmoid(self.zs[-1], True))
        biases_copy = np.copy(self.biases)
        weights_copy = np.copy(self.weights)
        biases_copy[-1] = delta
        nodes_2 = np.matrix(self.nodes[-2])
        weights_copy[-1] = np.dot(delta, nodes_2.transpose())
        # Applying back propagation to the rest of the nodes.
        for back in range(1, self.layers - 1):
            weight = np.matrix(self.weights[-back])
            delta = np.multiply(np.dot(weight.transpose(), delta), sigmoid(self.zs[-back-1], True))
            biases_copy[-back-1] = delta
            nodes = np.matrix(self.nodes[-back-2])
            weights_copy[-back-1] = np.dot(delta, nodes.transpose())

        return biases_copy, weights_copy

    def collect_start(self):
        """ Collects the network structure, biases, and weights of the network and stores it in a cvs file.
        """
        with open("Starting Data.csv", "w") as f:
            f.write("Network structure, " + ", ".join(str(size) for size in self.sizes) + "\n")
            f.write("Biases, " + ", ".join(str(biases.transpose()[0]) for biases in self.biases))
            for layer in range(self.layers - 1):
                f.write("\nWeights for layer " + str(layer) + ", "
                        + ", ".join(str(self.weights[layer][i]) for i in range(self.sizes[layer] + 1)))
            f.close()

    def collect_all(self):
        """ Collects the network structure, biases, weights, and nodes of the networks and stores it in a csv file.
        """
        with open("All Data.csv", "w") as f:
            f.write("Network structure, " + ", ".join(str(size) for size in self.sizes) + "\n")
            f.write("Biases, " + ", ".join(str(biases.transpose()[0]) for biases in self.biases))
            for layer in range(self.layers - 1):
                f.write("\nWeights for layer " + str(layer) + ", "
                        + ", ".join(str(self.weights[layer][i]) for i in range(self.sizes[layer] + 1)))
            f.write("\nNodes, " + ", ".join(str(np.array(node.transpose())[0]) for node in self.nodes))
            f.close()


# Other functions that are needed.
def sigmoid(x, prime=False):
    """ Applies the sigmoid/sigmoid prime function to every element of the np.matrix.
    :param x: The input of the np.matrix to apply sigmoid/sigmoid prime to.
    :param prime: Prime=True will apply the sigmoid prime to x.
    :return: This will return a np.matrix of sigmoid/sigmoid prime function applied to every element of the np.matrix.
    """
    if prime:
        return np.multiply(sigmoid(x), (1-sigmoid(x)))
    return 1 / (1 + np.exp(-x))


def cost(self, correct_data, prime=False):
    """ Works out the cost/cost prime of each element of the output of the network
    :param self: This is used to get the get the last layer of the nodes of the network
    :param correct_data: The output value as a np.matrix column of the correct value for the input.
    :param prime: Prime=True will work out the cost prime for each element of the output of the netowork
    :return: This will return a np.matrix of cost/cost prime function applied to every element of the np.matrix.
    """
    if prime:
        return 2 * np.subtract(self.nodes[-1], correct_data)
    return np.sum(np.square(np.subtract(self.nodes[self.layers - 1], correct_data)))

<<<<<<< HEAD
if __name__ == "__main__":
    "This is used for testing of the Network and has nothing to do with the networks functionality."
    A = Network([2,3,1])
    A.feed_forward([1,2])
    print(A.weights)
    print(A.biases)
    print(A.nodes)
    A.back_prop([1,2],[1])

=======
>>>>>>> 019f9c0084b93b9c0a9bf5a6b4884b370e960df8
