import numpy as np

class Layer_Dropout:
    def __init__(self, rate=0.2):
        self.rate = rate
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        
        if not training:
            self.output = inputs
            return
            
        # Generate binary mask
        self.binary_mask = np.random.binomial(1, 1-self.rate, size=inputs.shape) / (1-self.rate)
        self.output = inputs * self.binary_mask
        
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask 