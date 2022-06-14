import numpy as np
from numba import jit, cuda

# @jit(nopython=True)
def yield_mask(input, window_size, window_stride):
        # if input.ndim > 3:
        #     input = np.squeeze(input)
        # if input.ndim == 2:
        #     (height, width) = input.shape
        #     channels = 1
        # else:
        (height, width, channels) = input.shape
            
        for i in range(0, height, window_stride[0]):
            if i + window_size[0] > height:
                break
            for j in range(0, width, window_stride[1]):
                if j + window_size[1] > width:
                    break
                for k in range(0, channels):
                    mask = input[i:i+window_size[0], j:j+window_size[1], k] #change to filter_size[]
                    yield mask, i, j, k