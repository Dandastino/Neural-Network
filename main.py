import numpy as np

np.random.seed(0)

X = [[3.4 , 2.1, 5.6, 4.39],
         [1.2, 3.4, 2.1, 0.5],
         [0.5, 1.2, 3.4, 2.1]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(1, n_neurons)
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
