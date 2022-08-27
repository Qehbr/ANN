# **ANN - Artifical Neural Network**
**General** ANN with many settings
In main MNIST Digit Recognition is implemented, but ANN can be used for many purposes
There are not convolutional layers implementation, MNIST Digit recognition got **97.21%** accuracy.

Files and classes:

**network.py** - class containing ANN class with functions:
  1. __init__ - constructor
  2. fit - learning+testing of ANN
  3. test - testing of ANN
  4. softmax - applies softmax on layer
  
**activation_function.py** - abstract class of activation function specified in layer:
  1. ReLU
  2. Sigmoid
  3. Tanh
  You can add activation functions by inheritating from activation_function class
  
**cost_function.py** - abstract class of cost function of ANN:
  1. Squared Sum
  2. Cross Entropy
  You can add cost functions by inheritating from cost_function class

MNIST Digit Recognition Part:

**mnist.pkl.gz** - contains MNIST data

**mnist_loader.py** - script for loading MNIST data

**main.py** - creating, training, and testing ANN for MNIST Digit recognition




  
