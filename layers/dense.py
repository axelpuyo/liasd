# import time
import numpy as np
from layers.layer import Layer

class Dense(Layer):
    def __init__(self, output_shape):
        if hasattr(output_shape, "__len__"):
            self.output_shape = output_shape
        else:
            self.output_shape = (output_shape,)
    
    def infer_shape(self, input_shape):
        self.input_shape = input_shape
        self.weights = np.random.rand(*self.output_shape, *self.input_shape) / (int(self.output_shape[0]) * int(self.input_shape[0]))
        self.bias = np.zeros((*self.output_shape, 1))
        return self.output_shape

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
        
        return input_gradient
