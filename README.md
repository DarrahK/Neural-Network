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

We will use the notation [3, 4, 2] to express this Network. Where the 3 is the inputs nodes, 4 is the hidden layer, and 2 is the output nodes.

### Initiations A Network Class
Parameters:
* layer_s - Array or Int. If an array is passed it will built a network with the desired layers in the array. If an Int is passed it will build a network only consisting of an input node layer.
* act_function - Default Sigmoid. If the user has a preference on the activation function it will use the given activation function for each layer of the network. 
* a - Default None. If the activation function requires an a value like RPeLU Or ELU you can pass the a value in initations.

## Built With
* [NumPy](http://www.numpy.org/) - The maths framework used

### Examples of building a Network.

```
  network = Network([3, 4, 2])
```

## More customisation of the network.
### Modular Network
```
  Network = Network(3)
  Network.add_layer(4, "RPeLU", 5)
  Network.add_layer(2, "ArcTan")
```
This creates a network with 3 inputs, 1 hidden layer of with 4 nodes, and 2 output nodes. We will use the notation [3, 4, 2] to express this Neural Network. As you can see we have used different activation function for each layer this allows the user to have more control over the network and can work with different combinations to achieve different results.


## P.S.
This my first ever project and I learnt Python 2 months ago. Please be nice. <3
