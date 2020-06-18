# neural-network-2.0 --- NOT FINISHED
My motivation was to allow beginners to machine learning to have an application that they can use to learn and experiment with. Where the user can create, train, and run a network in a couple of lines. Whilst offering more advance options like modularity and customization over activation function on different layers. To come up with new and innovative solutions to solve real problems.
 
## Creating a Neural Network

A neural network is built on the idea of input, hidden, and output layers.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/296px-Colored_neural_network.svg.png)

[Source](https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg) - Colored neural network

We will use the notation [3, 4, 2] to express this Network. Where the 3 how many input values, 4 is the number of nodes in the single hidden layer, and 2 is how many outputs nodes.

## Initiations A Network Class
```python
network(layer_s, act_func = "Sigmoid", a_const = None)
```
Parameters:

* layer_s : Array or Int 
  * [Type : Array] If an array is passed it will built a network with the desired layers in the array
  * [Type : Int]  If an Int is passed it will build a network only consisting of only an input node layer.

* act_funct : String 
  * Default "Sigmoid". If the user has a preference on the activation function it will use the given activation function for each layer of the network.

* a_const : Int
  *  Default None. If the activation function requires an a value like RPeLU Or ELU you can pass the a value in.

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
network.add_layer(num_of_nodes, act_func = "Sigmoid", a_const = None)
```
This method adds an another layer to the neural network with the activation function specified.

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
network.change_layer(layer, num_of_nodes)
```
This method changes how many nodes are on a layer.

Parameters:

* layer : Int. 
  * The layer that you want to change " remember the layers start from 0 "

* numbers_of_nodes :  Int 
  * The number of nodes you want on that layer.

```python
# This will change the second layer nodes to 5.

network.change_layer(1, 5)
```
---
```python
network.feed_forward(input_data)
```
This method will feed the given a numpy array into the network by feed forward algorithm. [ ADD LINK ] [FIX WORDING]

Parameters:

* input_data : numpy.array 
  * This is the input that you want to feed into the Network. [FIX WORD]

```python
# This will feed forward the array [1, 1, 1] into the Network.

network.feed_forward(numpy.array([[1], [1], [1]]))
```
---
```python
network.back_propagation(input_data, y, update = False, learning_rate = 0.05)
```
[ADD FUNCTIONALITY]

Parameters:

* input_data : numpy.array
  *  This is the given input for the Network so we can feed it forward.

* y : numpy.array
  * This is the correct output for the given input so we can apply backpropagation.

* update : Bool 
  * Default - False. If update is set to True it will update the weights and biases of the Network.

* learning_rate : Float
  * [ADD STUFF HERE] 
```python
# This will apply back propigation to the Network.

network.back_prop(numpy.array([[1], [1], [1]]), numpy.array([[1], [0]]))
```
---
 ```python
 network.SGD(input_data, epochs = 1, learning_rate = 0.05, training_data = None)
```
[ADD FUNCTIONALITY]
Parameters:
* input_data : numpy.array
  * -- 
* epochs : Int
  * --
* num_batchs : Int
  * --
* learning_rate : Float
  * --
* training_data : numpty.array
  * --
```python
add example
```
---
```python
network.mini_batch_SDG(data, epochs = 1, num_batchs = 4, learning_rate  = 0.05, training_data = None)
```   
[ADD FUNCTIONALITY]
Parameters:
* data : numpy.array
  * --
* epochs : Int
  * --
* num_batchs : Int
  * --
* learning_rate : Float
  * --
* training_data : np.array
  * --
```python
code sample
```
---
```python
network.score(training_data)
```
[ADD FUNCTIONALITY]

* training_data : np.array
  * --
```python
code sample
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
Here is a list of all the cost functions that are already built-in and links to understand what each of them do. You are able to add an activation function in the `.cost_functions.py` file, just make sure you add [ADD STUFF].

* [Quadratic Loss Function](https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function)

## Saving & Loading Models

Pickling

## Preparing Data

class

## Example of building Networks - Could change this so it gives examples of how to make, run and train

### Linearly

```python
# Initiations using an array to build a Neural network using the default activation function Sigmoid.

network = Network([3, 4, 2])

# Initiations using an array to build a Neural network using Tanh as the activation function.

network = Network([3, 4, 2], "Tanh")
```

### Modular

```
Network = Network(3)
Network.add_layer(4, "RPeLU", 5)
Network.add_layer(2, "ArcTan")
```

This creates a [3, 4, 2] network with differnt activation function for each layer, allowing the user to have more control over the network. This can work with different combinations to achieve different results.

## Case Study

MIST data set - Save it inside the here.
  
## Dependency

* [NumPy](http://www.numpy.org/) 
