import numpy as np

class softmaxLayer:
    def __init__(self, n_in):
        self.num_inputs = n_in
        self.num_outputs = n_in
    
    def softmax(self, x):
        y = np.exp(x - np.max(x))
        z = y/np.sum(y)
        return x, y, z
    
    def forward(self, input):
        if input.ndim > 1:
            input.flatten()

        self.input_size = len(input)
        self.x, self.y, self.output = self.softmax(input)
        # print('Prediction Max : ', np.max(self.output), 'Prediction Min : ', np.min(self.output))
        return self.output

    def backward(self, grad0, *args):
        # z = self.output
        # jacobian = z * np.identity(z.size) - z.transpose() @ z
        # dL_dX = grad0 @ jacobian
        n = np.size(self.output)
        tmp = np.tile(self.output, n).reshape((n,n))
        dL_dX = np.dot(tmp * (np.identity(n) - np.transpose(tmp)), grad0)
        # print(grad0, dL_dX)
        return dL_dX
        # for i, grad in enumerate(grad0): # On veut grad_out[i] et i.
        #     if grad == 0:
        #         continue
        #     # grad = dL/dY, let's backpropagate.
        #     expX = np.exp(self.x)
        #     S = np.sum(expX)
        #     # print(self.Z)
        #     dY_dX = - expX[i]*expX/(S**2) # cas général
        #     dY_dX[i] = expX[i]*(S - expX[i]) / (S**2) # cas où on cherche le gradient de Yi par rapport à Zi

        #     dL_dX = grad*dY_dX #dL_dZ = dL_dY * dY_dZ

        #     # np.array 1D en vecteur ligne et vecteur colonne.
            
        # return dL_dX.reshape(self.input_size)
