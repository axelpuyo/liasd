import numpy as np
import scipy as sp

class convolutionalLayer: 
    def __init__(self, convolution_type, filter_size, padding, stride): # Convention : height x width x channels
        self.type = convolution_type
        self.filter_height = filter_size[0]
        self.filter_width = filter_size[1]
        self.num_channels = filter_size[2]
        self.filter = np.random.randn(self.filter_height, self.filter_width, self.num_channels)/(self.filter_height*self.filter_width)
        self.padding = padding
        self.stride = stride

    def set_filters(self, filters):
        self.filter_height = filters.shape[0]  # Convention : height x width x channels
        self.filter_width = filters.shape[1]
        self.num_channels = filters.shape[2]
        self.filter = filters

    def convolution_operation(self, A, B): # numpy arrays
        if self.type == 'default':
            return np.sum(A*B, axis = (0,1))
        else:
            raise('Define a valid convolution operation.')

    def forward(self, input_image):
        self.input_size = input_image.shape
        self.output_size = ((self.input_size[0] - self.filter_height + self.padding[0] + 1)//self.stride[0], (self.input_size[1] - self.filter_width + self.padding[1] + 1)//self.stride[0], self.num_channels) 
        output_image = np.zeros(self.output_size)

        for i in range(0, self.output_size[0], self.stride[0]):
            for j in range(0,self.output_size[1], self.stride[1]):
                    A = input_image[i:i+self.filter_height, j:j+self.filter_width,:]
                    B = self.filter
                    output_image[i,j,:] = self.convolution_operation(A, B)
        
        self.input_image = input_image # Utile pour la backpropagation
        return output_image


    def backward(self, gradOut, learning_rate):
        gradInp = np.zeros(self.filter.shape)
        for i in range(0, self.output_size[0], self.stride[0]):
            for j in range(0, self.output_size[1], self.stride[1]):
                for k in range(self.num_channels):
                    gradInp[:,:,k] += self.input_image[i:i+self.filter_height, j:j+self.filter_width]*gradOut[i, j, k]
        
        self.filter -= learning_rate*gradInp 
        
        return gradInp