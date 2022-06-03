import numpy as np

class poolingLayer:
    def __init__(self, operation, padding, mask_size, stride, axis):
        ''' 
        Initializes a pooling layer.

        :operation: 'Max', 'Avg' or 'Min' are the valid pooling operations.
        :padding: 'valid' will decrease the dimension of the outputs, 'same' will add the necessary padding to keep them the same as the input size.
        :mask_size: Size of the pooling mask.
        :stride: Stride of the pooling mask.
        '''
        self.operation = operation
        self.padding = padding
        self.mask_size = mask_size
        self.stride = stride
        self.pool_dims = axis 
    
    def mask_generator(self, input):
        (height, width, num_channels) = self.input.shape
        for i in range(0, height, self.stride[0]):
            if i + self.mask_size[0] > height:
                break
            for j in range(0, width, self.stride[1]):
                if j + self.mask_size[1] > width:
                    break
                for k in range(num_channels):
                    mask = input[i:i+self.mask_size[0], j:j+self.mask_size[1], k]
                    yield mask, i, j, k # Generator (we only need one patch at a time) : outputs the image patch, and its starting coordinates.

    def pool(self, mask):
        if self.operation == 'Max':
            return np.amax(mask, axis = self.pool_dims)
        elif self.operation == 'Min':
            return np.amin(mask, axis = self.pool_dims)
        elif self.operation == 'Avg':
            return np.average(mask, axis = self.pool_dims)
        else:
            raise('Define a valid pooling operation.')

    def initialize_weights(self):
        pass
    
    def forward(self, input):
        self.input = input # Storing the original image will be useful for backpropagation
        output_height = (input.shape[0] - self.mask_size[0] + 2*self.padding[0])//self.stride[0] + 1 # Define how many times the filter will go through the matrix
        output_width = (input.shape[1] - self.mask_size[1] + 2*self.padding[1])//self.stride[1] + 1
        num_channels = input.shape[2]

        self.output_size = (output_height, output_width, num_channels)
        self.input_size = input.shape
        out = np.zeros(self.output_size)

        for mask, i, j, k in self.mask_generator(input):
            out[i//self.stride[0], j//self.stride[1], k] = self.pool(mask)
        
        self.output_image = out
        return self.output_image
    
    def backward(self, grad0, *args): # This isn't taking into account the stride.
        '''
        :grad0: previous gradient.
        '''
        from utils import mask_generator
        grad = np.zeros(self.input.shape)
        for mask, i, j, k in mask_generator.yield_mask(self.input, self.mask_size, self.stride):
            pooled_value = self.output_image[i//self.stride[0], j//self.stride[1], k]

            for x in range(self.mask_size[0]):
                for y in range(self.mask_size[1]):
                    if self.input[i + x, j + y, k] == pooled_value:
                        grad[i + x, j + y, k] = grad0[i//self.stride[0], j//self.stride[1], k]
        
        return grad
        
