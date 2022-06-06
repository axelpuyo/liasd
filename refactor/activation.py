import numpy as np
from layers.layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_deriv):
        self.activation = activation
        self.activation_deriv = activation_deriv

    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, grad0, *args):
        return np.multiply(grad0, self.activation_deriv(self.input))

