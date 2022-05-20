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
                for k in range(self.num_channels):
                    A = input_image[i:i+self.filter_height, j:j+self.filter_width]
                    B = self.filter[:,:,k]
                    output_image[i,j,k] = self.convolution_operation(A, B)
        
        self.input_image = input_image # Utile pour la backpropagation
        return output_image


    def backward(self, gradOut, learning_rate):
        gradInp = np.zeros(self.filter.shape)
        for i in range(0, self.output_size[0], self.stride[0]):
            for j in range(0, self.output_size[1], self.stride[1]):
                for k in range(self.num_channels):
                    gradInp[:,:,k] += self.input_image[i:i+self.filter_height, j:j+self.filter_width]*gradOut[i, j, k]
        
        self.filter = self.filter - learning_rate*gradInp # seems not to be getting updated
        return gradInp

    # def __init__(self, f, size, stride):
    #     self.num_filters = f
    #     self.filter_size = size
    #     self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size**2) # Why are we normalizing by filter_size**2 ?

    # def set_weights(self, weights): # In case we want to set custom filters or have a pre-trained network.
    #     self.num_filters = weights.shape[0]
    #     self.filter_size = weights.shape[1]
    #     self.conv_filter = weights

    # def image_region(self, image):
    #     height, width = image.shape # Assumes single-channel image
    #     self.image = image
    #     for i in range(height - self.filter_size + 1): # Assumes vertical stride = 1
    #         for j in range(width - self.filter_size + 1): # Assumes horizontal stride = 1
    #             image_patch = image[i:(i + self.filter_size), j:(j + self.filter_size)]
    #             yield image_patch, i, j # Generator
    
    # def forward_prop(self, image): 
    #     height, width = image.shape # Assumes single-channel image
    #     conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters)) # Assumes no padding (valid), stride = 1,  single channel image, 
    #     for image_patch, i, j in self.image_region(image):
    #         conv_out[i,j] = np.sum(image_patch*self.conv_filter, axis = (1,2)) # Assumes basic 2D convolution
    #     return conv_out

    # def back_prop(self, dL_dout, learning_rate):
    #     '''
        
    #     :param dL_dout: Comes from Pooling unit
    #     '''
    #     dL_dinp = np.zeros(self.conv_filter.shape)
    #     for image_patch, i, j in self.image_region(self.image):
    #         for k in range(self.num_filters):
    #             dL_dinp[k] += image_patch*dL_dout[i, j, k]
        
    #     self.conv_filter -= learning_rate*dL_dinp
    #     return dL_dinp