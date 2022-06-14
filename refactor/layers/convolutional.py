import numpy as np
import time

from layers.layer import Layer
from utils.convolutions import convolution

class Convolutional(Layer):
    def __init__(self, conv_type, kernel_shape, padding, stride):
        self.convolution_type = conv_type
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride

        # self.kernels = np.ones((kernel_shape))
        self.kernels = np.random.randn(*self.kernel_shape) / (self.kernel_shape[0] * self.kernel_shape[1])

    def forward(self, input):
        self.input = input
        self.input_shape = input.shape
        self.output_shape = ((self.input_shape[0] - self.kernel_shape[0] + 2*self.padding[0] )//self.stride[0] + 1, (self.input_shape[1] - self.kernel_shape[1] + 2*self.padding[1])//self.stride[0] + 1, self.kernel_shape[2]) 
        self.bias = np.zeros((self.output_shape))

        self.output = convolution(input, self.kernels, self.padding, self.stride, self.convolution_type) + self.bias # Z = conv(K, X) + B
        return self.output
    
    def backward(self, output_gradient, lr):
        kernels_gradient = np.squeeze(convolution(self.input, output_gradient, self.padding, self.stride, self.convolution_type))

        padded = np.pad(output_gradient, pad_width = self.kernel_shape[0] - 1)
        flipped = np.fliplr(np.flipud(self.kernels))
        input_gradient = np.squeeze(convolution(padded, flipped, self.padding, self.stride, self.convolution_type))

        self.kernels -= lr * kernels_gradient
        self.bias -= lr * output_gradient
        return input_gradient
       
    
