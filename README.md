
# neural-network-2.0

My motivation was to allow beginners to machine learning to have an application which they can use to learn and experiment with. This project allows people to create, train, and run a network in a couple of lines, whilst offering more advance options like modularity and customization over activation function on different layers. This allows. people to come up with new and innovative solutions to solve problems.
 
## Creating a Neural Network

A neural network is built on the idea of input, hidden, and output layers.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/296px-Colored_neural_network.svg.png)

[Source](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg) - Colored neural network

We will use the notation [3, 4, 2] to express this network. Where the 3 how many input values, 4 is the number of nodes in the single hidden layer, and 2 is how many outputs nodes.

## Initiations A Network Class
```python
Network(layer_s, act_func = "Sigmoid", a_const = None) -> Object
```
Initialises the neural network class in the configuration that you want.

Parameters:

* layer_s : Array or Int 
  * [Type : Array] If an array is passed, it will builds a network with the desired layers of the array.
  * [Type : Int]  If an Int is passed, it will builds a network only consisting of an input node layer.

* act_funct : String 
  * Default - "Sigmoid". If the user has a preference of the activation function it will use the that activation function for each layer of the network.

* a_const : Int
  *  Default - None. If the activation function requires a value like RPeLU Or ELU.

### Examples of Initiations of a Network.

```python
# Initiations using an Array to build a neural network using Tanh as the activation function.

network = Network([3, 4, 2], "Tanh")
```

```python
# Initiations using an Int to build a Neural network with only the input node layer with default activation function Sigmoid.

network = Network(3)
```

### Methods
```python
network.add_layer(num_of_nodes, act_func = "Sigmoid", a_const = None) -> None
```
This method adds an another layer to the end of the neural network with the activation function specified.

Parameters:

* num_of_nodes : Int
  *  How many nodes do you want to add to the network.

* act_func : String
  * Default - "Sigmoid". The activation function to be associated with that layer.

* a_const - Int
  *  Default - None. If the activation function requires an a value for activation functions like RPeLU Or ELU.

```python
# This will add another layer with 3 nodes to the Network with the default Sigmoid activation function.

network.add_layer(3)
```

```python
# This will add another layer with 4 nodes to the Network with the RPeLU activation function.

network.add_layer(4, "RPelU", 0.3)
```
-----
```python
Network.change_layer(layer, num_of_nodes) -> None
```
This method changes how many nodes are on a layer starting from 0 to length of the network - 1.

Parameters:

* layer : Int. 
  * The layer that you want to change.

* numbers_of_nodes :  Int 
  * The number of nodes you want on that layer.

```python
# This will change the second layer nodes to 5.

network.change_layer(1, 5)
```
---
```python
Network.feed_forward(input_data) -> numpy.array
```
This method will feed the given the numpy array into the network using the feed forward algorithm. [ READ MORE ](https://towardsdatascience.com/deep-learning-feedforward-neural-network-26a6705dbdc7) 

Parameters:

* input_data : numpy.array 

  * This is the input that you want to feed into the Network. 

```python
# This will feed forward the array [1, 1, 1] into the Network.

network.feed_forward(numpy.array(numpy.array([[1], [1], [1]])))
```
---
```python
Network.back_propagation(input_data, y, update = False, learning_rate = 0.05) -> [(numpy.array, numpy.arry)] | None
```
This method will feed the given the numpy array into the network and apply back propagation to it. [READ MORE](https://en.wikipedia.org/wiki/Backpropagation)

Parameters:

* input_data : numpy.array
  *  This is the given input for the network so we can feed it forward.

* y : numpy.array
  * This is the correct output for the given input so we can apply backpropagation.

* update : Bool 
  * Default - False. If update is set to True, it will update the weights and biases of the network.

* learning_rate : Float
  * This will update the weights and biases with the given learning rate.
```python
# This will apply back propigation to the Network.

network.back_prop(numpy.array([[1], [1], [1]]), numpy.array([[1], [0]]))
```
---
 ```python
 Network.SGD(input_data, epochs = 1, learning_rate = 0.05, training_data = None) -> None
```
This method will feed the given a tuple of input and out as a list into the network and apply stochastic gradient descent to the network. [READ MORE](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

Parameters:
* input_data : [(numpy.array, numpy.array)]
  * list of tuples if with the first element of the tuple being the input and the second item in the tuple being the output.
* epochs : Int
  * Default - 1. How many times you want to use the training data. 
* learning_rate : Float
  * Default - 0.05. This will update the weights and biases with the given learning rate.

* training_data : [(numpy.array, numpy.array)]
  * list of tuples if with the first element of the tuple being the input and the second item in the tuple being the output.
```python
network.SGD(data, 6, 0.01, training_data)
```
---
```python
Network.mini_batch_SGD(data, epochs = 1, num_batchs = 4, learning_rate  = 0.05, training_data = None) -> None
```   
This method will feed the given a tuple of input and out as a list into the network and apply mini batch stochastic gradient descent to the network. [READ MORE](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a)

Parameters:
* data : [(numpy.array, numpy.array)]
  * list of tuples if with the first element of the tuple being the input and the second item in the tuple being the output.
* epochs : Int
  * Default - 1. How many times you want to use the training data. 
* num_batchs : Int
  * Default - 4. How much do you want to split up your training data set.
* learning_rate : Float
  * Default - 0.05. This will update the weights and biases with the given learning rate.

* training_data : [(numpy.array, numpy.array)]
  * list of tuples if with the first element of the tuple being the input and the second item in the tuple being the output.
```python
network.min_batch_SDG(data, 4, 3, 0.01)
```
---
```python
Network.score(training_data)
```
Works out the average cost of all the training data.

* training_data : [(numpy.array, numpy.array)]

  * list of tuples if with the first element of the tuple being the input and the second item in the tuple being the output.
```python
costs = network.score(data)
```

## Activation functions:
Here is a list of all the activation functions that are already built-in and links to understand what each of them do. You are able to add an activation function in the `.activation_functions.py` file, just make sure you add [ADD STUFF].

* [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)

* [Tanh](http://reference.wolfram.com/language/ref/Tanh.html)

* [RelU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

* [AcrTan](http://reference.wolfram.com/language/ref/ArcTan.html)

* [ID](https://en.wikipedia.org/wiki/Identity_function)

* [RPeLU](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions)

* [ELU](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/)

* [SolfPlus](https://sefiks.com/2017/08/11/softplus-as-a-neural-networks-activation-function/)

## Cost function
Here is a list of all the cost functions that are already built-in and links to understand what each of them do. You are able to add an activation function in the `.cost_functions.py` file.

* [Quadratic Loss Function](https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function)

## Preparing Data

When preparing data, make sure the csv file takes the form of inputs first and then outputs after on the same line. The load data function will return a list of tuples if with the first element of the tuple being the input and the second item in the tuple being the output.

```python
load_data(file_name, num_of_inputs) -> [(numpy.array, numpy.array)]
```
Loads data used to back propagation.

Parameters:

* file_name : String
  * Name of the csv file that you want to load.

* num_of_inputs : Int
  *  The number of inputs nodes that you have in your neural network.

## Saving & Loading Models

```python
save(file_name, object) -> None
```

Saves the network model into memory

Parameters:

* file_name : String
  * Name of the file that object ( neural network ) to be saved to.

* object : Class
  *  Variable that the neural network is assigned to.

--- 
```python
load(file_name) -> Object
```

Loads the network that you were already using.

Parameters:

* file_name : String
  * Name of the file that object ( neural network ) to be saved to.


## Example of building Networks - Could change this so it gives examples of how to make, run and train

### Linearly

```python
# Initiations using an array to build a Neural network using the default activation function Sigmoid.

network = Network([3, 4, 2])

# Initiations using an array to build a Neural network using Tanh as the activation function.

network = Network([3, 4, 2], "Tanh")
```

### Modular

```python
Network = Network(3)
Network.add_layer(4, "RPeLU", 5)
Network.add_layer(2, "ArcTan")
```

This creates a [3, 4, 2] network with differnt activation function for each layer, allowing the user to have more control over the network. This can work with different combinations to achieve different results.

### Example of Building, Training, And Saving

```python
network = Network([3,4,2])
training_data = load_data("taining_data.csv", 3)
network.SGD( training_data, 10, 0.01)
save("network.obj", network)
```  

## Dependency

* [NumPy](http://www.numpy.org/) 

## Comments

The aim of project was for me to have an introduction to Python, Git, OOP, and machine learning. I followed the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) closely through the project, learning the Mathematics of machine learning along the way.

If there is any problems will the code, please message me :) 
