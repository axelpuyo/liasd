import numpy as np
from layers.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.ones((output_size, input_size)) / (output_size * input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input.flatten()[np.newaxis]
        return self.weights @ self.input.T + self.bias
    
    def backward(self, grad0, learning_rate):
        if grad0.ndim < 2:
            grad0 = grad0[np.newaxis]
        elif grad0.ndim > 2:
            grad0 = np.squeeze(grad0)
        
        weights_gradient = grad0.T @ self.input
        input_gradient =  grad0 @ self.weights
        bias_gradient = grad0.T
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        # print(input_gradient.shape)
        return input_gradient
