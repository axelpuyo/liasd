import numpy as np
from layers.layer import Layer

class Reshape(Layer):
    def __init__(self):
        pass
    
    def infer_shape(self, input_shape):
        self.output_shape = 1
        self.input_shape = input_shape
        for dim in input_shape:
            self.output_shape *= dim
        
        self.output_shape = (self.output_shape, ) # Dangereux si jamais appelé plusieurs fois de suite.
        
        return self.output_shape

    def forward(self, input):
        self.output = input.flatten()
        return input.flatten()

    def backward(self, output_gradient, *args):
        return np.reshape(output_gradient, self.input_shape)
