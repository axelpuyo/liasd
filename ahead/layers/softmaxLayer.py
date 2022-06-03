import numpy as np

class softmaxLayer:
    def __init__(self, n_in):
        self.num_inputs = n_in
        self.num_outputs = n_in
    
    def softmax(self, x):
        y = np.exp(x - np.max(x))
        z = y/np.sum(y, axis = 0)
        return x, y, z
    
    def forward(self, input):
        self.input_size = len(input)
        self.x, self.y, self.output = self.softmax(input)
        return self.output

    def backward(self, grad0, *args):
        z = self.output
        jacobian = z * np.identity(z.size) - z.transpose() @ z
        dL_dX = grad0 @ jacobian
        # print(grad0, dL_dX)
        return dL_dX
