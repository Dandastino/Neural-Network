import numpy as np

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  
        self.beta2 = beta2  
        self.epsilon = epsilon  
        self.m = {}  
        self.v = {}  
        self.m_b = {}  
        self.v_b = {} 
        self.t = 0 

    def update_params(self, layer):
        layer_id = id(layer)
        
        # Initialize momentum and velocity for weights if not already done
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(layer.weights)
            self.v[layer_id] = np.zeros_like(layer.weights)
            
        self.t += 1
        
        # Calculate momentum and velocity for weights
        self.m[layer_id] = self.beta1 * self.m[layer_id] + (1 - self.beta1) * layer.dweights
        self.v[layer_id] = self.beta2 * self.v[layer_id] + (1 - self.beta2) * layer.dweights**2
        
        # Bias correction for weights
        m_hat = self.m[layer_id] / (1 - self.beta1**self.t)
        v_hat = self.v[layer_id] / (1 - self.beta2**self.t)
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update biases if they exist
        if hasattr(layer, 'biases'):
            if layer_id not in self.m_b:
                self.m_b[layer_id] = np.zeros_like(layer.biases)
                self.v_b[layer_id] = np.zeros_like(layer.biases)
            
            # Calculate momentum and velocity for biases
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.dbiases
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * layer.dbiases**2
            
            # Bias correction for biases
            m_hat_b = self.m_b[layer_id] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[layer_id] / (1 - self.beta2**self.t)
            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon) 