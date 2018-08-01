# neural-network-1.0
Network.py was my first attempted of coding a Neural Network and still working 

# neural-network-2.0
The aim was to build to able to have the user have more control over the the Neural Network. This was done by adding modularity to the building of the Network. However I have also added a faster way to build a network if required.

## Current State of the project
Still in the early stages because of other commitments.

## Building a Network
To build a modular Network.
```
  Network = Network(3)
  Network.add_layer(4, "RelU")
  Network.add_layer(2, "ArcTan")
```
This creates a network with 3 inputs, 1 hidden layer of with 4 nodes, and 2 output nodes. We will use the notation [3, 4, 2] to express the network 
