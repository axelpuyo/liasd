import numpy as np
from layers.layer import Layer

class Convolutional(Layer):
    def __init__(self, conv_type, kernel_shape, padding, stride):
        self.convolution_type = conv_type
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride

        self.kernels = np.random.randn(*self.kernel_shape)
    
    def convolution(self, input, kernel):
        from utils import mask_generator
        output = np.zeros(self.output_shape)

        for mask, i, j, k in mask_generator.yield_mask(input, kernel.shape, self.stride): # mask_generator gère déjà le cas k = 1.
                output[i // self.stride[0], j // self.stride[1]] += np.sum(mask*kernel)

        return output

    def forward(self, input):
        self.input = input
        self.output_shape = ((input.shape[0] - self.kernel_shape[0]) // self.stride[0] + 1, (input.shape[1] - self.kernel_shape[1]) // self.stride[1] + 1, self.kernel_shape[2])
        self.bias = np.random.randn(self.output_shape)
        self.output = np.copy(self.bias)

        for i in range(self.kernel_shape[2]):
            self.output[..., i] = self.convolution(input, self.kernel)
        
        return self.output
    
    def backward(self, output_gradient, lr):
        
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.kernel_shape[2]):
            kernels_gradient[..., i] = self.convolution(self.input, output_gradient[..., n])
            input_gradient[...] = self.convolution(np.pad(output_gradient, pad_width = self.kernel_shape[0] -1), np.fliplr(np.flipud(self.kernel[..., i])))
        
        self.kernels -= lr * kernels_gradient
        self.bias -= lr * output_gradient
        return input_gradient

        # for i in range(self.depth):

        #     for j in range(self.input_depth):
        #         kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
        #         input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # self.kernels -= learning_rate * kernels_gradient
        # self.biases -= learning_rate * output_gradient
        # return input_gradient
    
