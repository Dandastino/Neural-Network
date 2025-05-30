import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


# Basic Layer for a neural network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(1, n_neurons)
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Classic Activation functions and layers for a neural network
class Activation_Linear:
    def forward(self, inputs):
        self.output = inputs

# Setting zero all the negative values in the output
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Solving the zero problem with the normalization after the exponentiation
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # for numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # normalize
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(np.array([[1, 2], [3, 4]]))

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5]) 
