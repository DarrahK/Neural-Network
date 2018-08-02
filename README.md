# neural-network-1.0
Network.py was my first attempted of coding a Neural Network and still working. 

# neural-network-2.0
The aim was to build to able to have the user have more control over the the Neural Network and allow beginners to machine learning to have an appication that they can use to learn and experiment. In each section there will be an explanation of what each parts of the network to get begineers familiar to Neural Networks and how they work.
This was done by adding modularity to the building of the Network. However I have also added a faster way to build a network if required.

## Current State of the project
Still in the early stages because of other commitments.

## Creating a Neural Network
A Neural Network is built on the idea of input, hidden and output layers.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/296px-Colored_neural_network.svg.png)

[Source](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg) - Colored neural network

Each layer is expressed as a column vector where each value is the value of the node from top to bottom. Each node has a biases
Each layer talks to each other in a linear fashion by a process called feed forward. This is achieved by a process of matrix 

as I have stated there is two ways to build a network modularly or on initiation. In both examples we will be creating A Nearul Network with 3 inputs, 1 hidden layer of with 4 nodes, and 2 output nodes. We will use the notation [3, 4, 2] to express this Network.



### Initiations Network
```
  Network = Network([3, 4, 2])
```
## Built With
* [NumPy](http://www.numpy.org/) - The maths framework used


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
