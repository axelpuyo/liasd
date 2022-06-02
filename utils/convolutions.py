def convolution_operation(X, K, stride, padding, type): # numpy arrays
    from utils import mask_generator
    import numpy as np
    
    if not K.shape[2]:
        num_filters = 1
    else:
        num_filters = K.shape[2]

    out = np.zeros(((X.shape[0] - K.shape[0] + padding[0] + 1)//stride[0], (X.shape[1] - K.shape[1] + padding[1] + 1)//stride[1], num_filters))
    if type == 'default':
        for n in range(num_filters):
            for mask, i, j, k in mask_generator.yield_mask(X, K.shape, stride): # mask_generator gère déjà le cas k = 1.
                out[i//stride[0], j//stride[1], n] += np.sum(mask*K[..., n])
            return out
    else:
        raise('Define a valid convolution operation.')