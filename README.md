# neural-network-1.0
Network.py was my first attempted of coding a Neural Network and still working. 

# neural-network-2.0
My motivation was to allow the user have more control over the the Neural Network and allow beginners to machine learning to have an appication that they can use to learn and experiment with. This will be done by allowing the user to create,run, and train a Network in a couple of simple lines. Whilst offering the options to modular with customization over activation function on different layers. To come up with different 

## Current State of the project
Still in the early stages because of other commitments.

## Creating a Neural Network
A Perceptron Neural Network is built on the idea of input, hidden and output layers.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/296px-Colored_neural_network.svg.png)

[Source](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg) - Colored neural network

We will use the notation [3, 4, 2] to express this Network. Where the 3 is the inputs nodes, 4 is the number of nodes in the single hidden layer, and 2 is the output nodes.

### Initiations A Network Class
Parameters:
* layer_s - tpye Array or Int. If an array is passed it will built a network with the desired layers in the array. If an Int is passed it will build a network only consisting of an input node layer.
* act_function - Type string: Default "Sigmoid". If the user has a preference on the activation function it will use the given activation function for each layer of the network. 
* a - type int: Default None. If the activation function requires an a value like RPeLU Or ELU you can pass the a value in initations.

 ### Examples of Initiations of a Network.

```
# Initiations using an array to build a Neural network using Tanh as the activation function.
network = Network([3, 4, 2], "Tanh")
```

```
# Initiations using an int to build a Neural network with only the input node layer.
network = Network(3)
```

### Methods
* add_layer
  * Parrameters:
    * Nodes - type int. 
    * activations - type string: Default "Sigmoid". The activation function to be associated with that layer.
    * a - type int: Default None. If the activation function requires an a value like RPeLU Or ELU.
  * Functionality:
    This method adds an another layer to the Neural Network with thee activation function specified.
```
# This will add another layer with 3 nodes to the Network with the default Sigmoid activation function.
network.add_layer(3)
# This will add another layer with 4 nodes to the Network with the RPeLU activation function.
network.add_layer(4, "RPelU", 0.3)
```
* change_layer
  * Parrameters:
    * layer - type int. The layer that you want to change " remember the layers start from 0 "
    * numbers_of_nodes - type int. The number of nodes you want on that layer.
  * Functionality:
    This method changes how many nodes are on a layer.
```
# This will change the second layer nodes to 5.
network.change_layer(1, 5)
```
* feed_forward
  * Parrameters:
    * input_data - type array. This is the input that you want to feed into the Network.
    * output - type Bool: Default False. If output is True it will output the output layer as a NumPy matrix object.
  * Functionality:
    This method will feed the given array into the Network by feedforward algorithm.
```
# This will feed forward the array [1, 1, 1] into the Network.
network.feed_forward([1, 1, 1])
```
* back_prop
  * Parrameters:
    * x - type array. This is the given input for the Network so we can feed it forward.
    * y - type array. This is the correct output for the given input so we can apply backpropagation.
    * update - type Bool: Default False. If update is set to True it will update the weights and biases of the Network.
  * Functionality:
    This method will feed the given array into the Network by feedforward algorithm.
```
# This will apply back propigation to the Network.
network.back_prop([1, 1, 1], [1, 0])
```

## Built With
* [NumPy](http://www.numpy.org/) - The maths framework used

### Activation functions:


## Example of building Networks
### Linearly 
```
# Initiations using an array to build a Neural network using the default activation function Sigmoid.
network = Network([3, 4, 2])
# Initiations using an array to build a Neural network using Tanh as the activation function.
network = Network([3, 4, 2], "Tanh")
```

### Modular Network
```
  Network = Network(3)
  Network.add_layer(4, "RPeLU", 5)
  Network.add_layer(2, "ArcTan")
```
This creates a network with 3 inputs, 1 hidden layer of with 4 nodes, and 2 output nodes with differnt activation function for each layer this allows the user to have more control over the network and can work with different combinations to achieve different results.


## P.S.
This my first ever project and I learnt Python 2 months ago. Please be nice. <3
