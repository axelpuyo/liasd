import numpy as np

class fullyConnectedLayer:
    def __init__(self, distribution, n_in, n_out):
        '''
        :distribution: non linear unit ('Softmax', 'ReLu', 'Elu')
        :n_in: number of input nodes (int)
        :n_out: number of output nodes (int)
        '''
        self.operation = distribution
        self.numInputs = n_in
        self.numOutputs = n_out
        self.weights = np.random.randn(self.numInputs, self.numOutputs)/self.numInputs
        self.bias = np.zeros(self.numOutputs)
        
    # def initialize_weights(self):
    #     self.weights = np.random.randn(self.numInputs, self.numOutputs)/self.numInputs
    #     self.bias = np.zeros(self.numOutputs)
        
    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias =  bias

    def soft(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z), axis = 0), np.exp(Z)
    
    def relu(Z):
        pass
    
    def forward(self, input):
        '''
        :input: square image input for now (numpy nd-array).
        '''
        self.input_size = input.shape
        self.flat = input.flatten()
                
        self.Z = np.dot(self.flat, self.weights) + self.bias # Z = W*X+B
        # if self.operation == 'Softmax':
        #     (self.prediction, self.expZ) = self.soft(self.Z)
        # elif self.operation == 'ReLu':
        #     pass
        # elif self.operation == 'Elu':
        #     pass
        # else:
        #     raise('Define a valid non-linear operation.')
        # return self.prediction
        return self.Z

    def backward(self, grad0, learning_rate):
        '''
        :grad_out: previous layer gradient (numpy ndarray)
        :learning_rate: learning rate (float)
        '''
        # for i, grad in enumerate(grad0): # On veut grad_out[i] et i.
        #     if grad == 0:
        #         continue
        #     # # grad = dL/dY, let's backpropagate.
        #     # expZ = np.exp(self.Z)
        #     # S = np.sum(expZ)
        #     # # print(self.Z)
        #     # dY_dZ = - expZ[i]*expZ/(S**2) # cas général
        #     # dY_dZ[i] = expZ[i]*(S - expZ[i]) / (S**2) # cas où on cherche le gradient de Yi par rapport à Zi

        dZ_dW = self.flat
        dZ_dX = self.weights # Z = W*X + B, X = input image en vecteur colonne.
        dZ_dB = 1

        dL_dZ = grad0 #dL_dZ = dL_dY * dY_dZ

        dL_dW = dZ_dW[np.newaxis].T @ dL_dZ[np.newaxis] # newaxis nécessaire pour convertir des 
        # np.array 1D en vecteur ligne et vecteur colonne.
        dL_dX = dZ_dX @ dL_dZ 
        dL_dB = dL_dZ * dZ_dB

            # Update parameters
        self.weights -= learning_rate*dL_dW
        self.bias -= learning_rate*dL_dB

        # print(np.max(self.flat))
            
        return dL_dX.reshape(self.input_size)