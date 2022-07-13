# import time
import numpy as np
from layers.layer import Layer
from utils.mask_generator import yield_mask

class Pooling(Layer):
    def __init__(self, mask_size, padding, stride):
        self.mask_size = mask_size
        self.padding = padding
        self.stride = stride
    
    def infer_shape(self, input_shape):
        h = (input_shape[0] - self.mask_size[0] + 2*self.padding[0])//self.stride[0] + 1
        w = (input_shape[0] - self.mask_size[0] + 2*self.padding[0])//self.stride[0] + 1
        d = input_shape[2]
        self.output_shape = (h,w,d)
        return self.output_shape

    def pool(self, input, *args):
        return np.amax(input)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        
        input = np.pad(input, pad_width = self.padding[0])
        for mask, i, j, k in yield_mask(input, self.mask_size, self.stride):
            self.output[i//self.stride[0], j//self.stride[1], k] = self.pool(mask)
        
        return self.output

    def backward(self, grad0, *args): # This isn't taking into account the stride.
        '''
        :grad0: previous gradient.
        '''
        grad = np.zeros(self.input.shape)
        for mask, i, j, k in yield_mask(self.input, self.mask_size, self.stride):
            pooled_value = self.output[i//self.stride[0], j//self.stride[1], k]

            for x in range(self.mask_size[0]):
                for y in range(self.mask_size[1]):
                    if self.input[i + x, j + y, k] == pooled_value:
                        grad[i + x, j + y, k] = grad0[i//self.stride[0], j//self.stride[1], k]
        return grad