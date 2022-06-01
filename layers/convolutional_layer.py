import numpy as np

class convolutionalLayer: 
    def __init__(self, convolution_type, filter_size, padding, stride): # Convention : height x width x channels
        '''
        :convolution_type: Convolution type (str) - 'Default' for Conv2D
        :filter_size: [fx, fy, fz] size of convolution window (int array)
        :padding: [px, py, pz] size of convolution padding (int array or s'valid for no padding, 'same' makes the output matrix keep the same size as the input)
        :stride: [sx, sy, sz] convolution window stride (int array)
        '''
        self.type = convolution_type
        self.filter_height = filter_size[0]
        self.filter_width = filter_size[1]
        self.num_filters = filter_size[2]
        self.filter = np.random.randn(self.filter_height, self.filter_width, self.num_filters)/(self.filter_height*self.filter_width)
        self.padding = padding
        self.stride = stride

    def set_filters(self, filters):
        self.filter_height = filters.shape[0]  # Convention : height x width x channels
        self.filter_width = filters.shape[1]
        self.num_filters = filters.shape[2]

        self.filter = filters

    def set_bias(self, bias):
        self.bias = bias
        
    def mask_generator(self, input):
            (height, width) = input.shape
            for i in range(0, height, self.stride[0]):
                if i + self.filter_height > height:
                    break
                for j in range(0, width, self.stride[1]):
                    if j + self.filter_width > width:
                        break
                        yield mask, i, j

    def convolution_operation(self, X, K): # numpy arrays
        out = np.zeros(((X.shape[0] - K.shape[0] + self.padding[0] + 1)//self.stride[0], (X.shape[1] - K.shape[1] + self.padding[1] + 1)//self.stride[1]))
        if self.type == 'default':
            for mask, i, j in self.mask_generator(X):
                out[i//self.stride[0], j//self.stride[1]] = np.sum(X*K)
            return out
        else:
            raise('Define a valid convolution operation.')

    def forward(self, input_image):
        self.input_size = input_image.shape
        self.output_size = ((self.input_size[0] - self.filter_height + self.padding[0] + 1)//self.stride[0], (self.input_size[1] - self.filter_width + self.padding[1] + 1)//self.stride[0], self.num_filters) 
        self.bias = np.zeros(self.output_size)
        output_image = np.zeros(self.output_size)
        for n in range(self.num_filters):
            K = self.filter[..., n]
            for k in range(input_image.shape[2]):
                 X = input_image[..., k]
                 output_image[..., n] += self.convolution_operation(X, K) # conv(K, X)
        
        output_image += self.bias # Z = conv(K, X) + B

        self.input_image = input_image # Utile pour la backpropagation
        return output_image


    def backward(self, grad0, learning_rate):
        
        dL_dK = np.zeros(self.filter.shape)
        dL_dB = np.zeros(self.output_size)
        dL_dX = np.zeros(self.input_size)
        for n in range(self.num_filters):
            for i in range(0, self.output_size[0], self.stride[0]):
                for j in range(0, self.output_size[1], self.stride[1]):
                    for k in range(self.input_size[-1]):
                        X = self.input_image[i:i+self.filter_height, j:j+self.filter_width, k]
                        dL_dK[..., n] += X*grad0[i, j, n]
            
            dL_dZ = np.pad(grad0[i:i+self.filter_height, j:j+self.filter_height, n], pad_width=1)
            K = np.fliplr(np.flipud(self.filter[..., n]))
            
            dL_dX[..., k] += self.convolution_operation(K, dL_dZ)
            
        self.filter -= learning_rate*dL_dK
        print(dL_dX.shape)
        return dL_dX