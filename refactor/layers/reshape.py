import numpy as np
from layers.layer import Layer

class Reshape(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()

    def backward(self, output_gradient, *args):
        return np.reshape(output_gradient, self.input_shape)
