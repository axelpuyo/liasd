from utils import mask_generator
import numpy as np

def convolution(input, kernel, padding, stride, type):
    if kernel.ndim > 2:
        num_filters = kernel.shape[2]
    else:
        num_filters = 1

    out = np.zeros(((input.shape[0] - kernel.shape[0] + padding[0] + 1)//stride[0], (input.shape[1] - kernel.shape[1] + padding[1] + 1)//stride[1], num_filters))
    if type == 'default':
        for n in range(num_filters):
            for mask, i, j, k in mask_generator.yield_mask(input, kernel.shape, stride): # mask_generator gère déjà le cas k = 1.
                out[i//stride[0], j//stride[1], n] += np.sum(mask * kernel[..., n])
        return out   
    else:
        raise('Define a valid convolution operation.')