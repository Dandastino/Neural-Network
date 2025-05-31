import numpy as np

class CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        return -np.log(correct_confidences)
    
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        self.dinputs = y_pred.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
