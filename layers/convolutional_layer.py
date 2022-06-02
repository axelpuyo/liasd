import numpy as np
from utils import convolutions

class convolutionalLayer: 
    def __init__(self, convolution_type, filter_size, padding, stride): # Convention : height x width x channels
        '''
        :convolution_type: Convolution type (str) - 'Default' for Conv2D
        :filter_size: [fx, fy, fz] size of convolution window (int array)
        :padding: [px, py, pz] size of convolution padding (int array or s'valid for no padding, 'same' makes the output matrix keep the same size as the input)
        :stride: [sx, sy, sz] convolution window stride (int array)
        '''
        self.type = convolution_type
        self.filter_size = filter_size
        self.num_filters = filter_size[2]
        self.filter = np.random.randn(self.filter_size[0], self.filter_size[1], self.num_filters)/(self.filter_size[0]*self.filter_size[1])
        self.padding = padding
        self.stride = stride

    def set_filters(self, filters):
        self.filter_size = filters.shape
        self.num_filters = filters.shape[2]
        self.filter = filters

    def set_bias(self, bias):
        self.bias = bias

    # def initialize_weights(self):
    #     self.filter = np.random.randn(self.filter_size[0], self.filter_size[1], self.num_filters)/(self.filter_size[0]*self.filter_size[1])

    def forward(self, input_image):
        self.input_size = input_image.shape
        self.output_size = ((self.input_size[0] - self.filter_size[0] + 2*self.padding[0] )//self.stride[0] + 1, (self.input_size[1] - self.filter_size[1] + 2*self.padding[1])//self.stride[0] + 1, self.num_filters) 
        self.bias = np.zeros(self.output_size)
        output_image = convolutions.convolution_operation(input_image, self.filter, self.stride, self.padding, self.type) + self.bias # Z = conv(K, X) + B
        
        self.input_image = input_image # Utile pour la backpropagation
        return output_image


    def backward(self, grad0, learning_rate):
        dL_dK = np.zeros(self.filter.shape)
        dL_dB = np.zeros(self.output_size)
        dL_dX = np.zeros(self.input_size)

        dL_dK = convolutions.convolution_operation(self.input_image, grad0, self.stride, self.padding, self.type)
        pad_size = (self.filter_size[0] - 1, self.filter_size[0] - 1)
        padded_grad = np.pad(grad0, pad_width = pad_size) # Implies square filters !!
        dL_dX = convolutions.convolution_operation(padded_grad, np.fliplr(np.flipud(self.filter)), self.stride, self.padding, self.type)

        # print('Input size :', grad0.shape, 'Output size :', dL_dX.shape)
        # print('Input non-zero :', grad0.any(), 'Output non-zero :', dL_dX.any())    
        self.filter -= learning_rate*dL_dK

        return dL_dX