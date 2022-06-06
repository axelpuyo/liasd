import numpy as np
from layers.layer import Layer
from activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(input):
            return 1 / (1 + np.exp(input))

        def sigmoid_deriv(input):
            s = sigmoid(input)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_deriv)
    
class ReLu(Activation):
    def __init__(self):
        def relu(input):
            return np.max(0, input)

        def relu_deriv(input):
            return 0 if input < 0 else 1
        
        super().__init__(relu, relu_deriv)

