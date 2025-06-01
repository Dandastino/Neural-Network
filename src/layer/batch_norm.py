import numpy as np

class Layer_BatchNorm:
    def __init__(self, momentum=0.9, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.gamma = None  
        self.beta = None 
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, inputs, training=True):
        self.input_shape = inputs.shape
        
        if self.gamma is None:
            self.gamma = np.ones(inputs.shape[1])
            self.beta = np.zeros(inputs.shape[1])
            
        if self.running_mean is None:
            self.running_mean = np.mean(inputs, axis=0)
            self.running_var = np.var(inputs, axis=0)
            
        if training:
            self.mean = np.mean(inputs, axis=0)
            self.var = np.var(inputs, axis=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            
            self.x_centered = inputs - self.mean
            self.x_normalized = self.x_centered / np.sqrt(self.var + self.epsilon)
        else:
            self.x_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            
        self.output = self.gamma * self.x_normalized + self.beta
        
    def backward(self, dvalues):
        batch_size = dvalues.shape[0]
        
        self.dgamma = np.sum(dvalues * self.x_normalized, axis=0)
        self.dbeta = np.sum(dvalues, axis=0)
        
        dx_normalized = dvalues * self.gamma
        
        dvar = np.sum(dx_normalized * self.x_centered * -0.5 * (self.var + self.epsilon)**(-1.5), axis=0)
        
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.var + self.epsilon), axis=0) + \
                dvar * np.sum(-2 * self.x_centered, axis=0) / batch_size
        
        self.dinputs = dx_normalized / np.sqrt(self.var + self.epsilon) + \
                      dvar * 2 * self.x_centered / batch_size + \
                      dmean / batch_size 