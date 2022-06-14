# import time
import numpy as np
from layers.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) / (output_size * input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        # start = time.time()
        self.input = input.flatten()[np.newaxis]
        # end = time.time()
        # print('forward dense -- time elapsed : ', end - start)
        # print(input.shape)
        return self.weights @ self.input.T + self.bias
        
    
    def backward(self, grad0, learning_rate):
        # start = time.time()
        if grad0.ndim < 2:
            grad0 = grad0[np.newaxis]
        elif grad0.ndim > 2:
            grad0 = np.squeeze(grad0)
        
        weights_gradient = grad0.T @ self.input
        input_gradient =  grad0 @ self.weights
        bias_gradient = grad0.T
        
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        # end = time.time()
        # print('backward dense -- time elapsed : ', end - start)
        return input_gradient
