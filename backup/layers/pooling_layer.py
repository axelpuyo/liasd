import numpy as np

class poolingLayer:
    def __init__(self, operation, mask_size, stride):
        self.operation = operation
        self.mask_size = mask_size
        self.stride = stride
    
    def mask_generator(self, input):
        (height, width, num_channels) = self.input.shape
        for i in range(0, height, self.stride[0]):
            for j in range(0, width, self.stride[1]):
                for k in range(num_channels):
                    mask = input[i:i+self.mask_size[0], j:j+self.mask_size[1], k]
                    yield mask, i, j, k # Generator (we only need one patch at a time) : outputs the image patch, and its starting coordinates.

    def pool(self, mask):
        if self.operation == 'Max':
            return np.amax(mask, axis = (0,1))
        elif self.operation == 'Min':
            return np.amin(mask, axis = (0,1))
        else:
            raise('Define a valid pooling operation.')

    def forward(self, input):
        self.input = input # Storing the original image will be useful for backpropagation
        new_height = (input.shape[0] - self.mask_size[0])//self.stride[0] + 1 # Define how many times the filter will go through the matrix
        new_width = (input.shape[1] - self.mask_size[1])//self.stride[1] + 1 # This assumes a stride of filter_size
        num_channels = input.shape[2]

        self.output_size = (new_height, new_width, num_channels)
        out = np.zeros(self.output_size)

        for mask, i, j, k in self.mask_generator(input):
            out[i//self.stride[0], j//self.stride[1], k] = self.pool(mask) # Max Pooling.
        
        self.out = out
        return self.out
    
    def backward(self, grad0): # This isn't taking into account the stride.
        '''
        :grad0: previous gradient.
        '''
        grad = np.zeros(self.input.shape)
        for mask, i, j, k in self.mask_generator(self.input):
            maxVal = self.out[i//self.stride[0], j//self.stride[1], k] #np.amax(mask, axis = (0,1))

            for x in range(self.mask_size[0]):
                for y in range(self.mask_size[1]):
                    if self.input[i + x, j + y, k] == maxVal:
                        grad[i + x, j + y, k] = grad0[i//self.stride[0], j//self.stride[1], k]
        
        return grad
        
