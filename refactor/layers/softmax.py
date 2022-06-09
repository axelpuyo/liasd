import numpy as np
from layers.layer import Layer

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, grad0, *args):
        """Computes the gradient of the softmax function.
        z: (T, 1) array of input values where the gradient is computed. T is the
        number of output classes.
        Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
        is DjSi - the partial derivative of Si w.r.t. input j.
        """
        Sz = self.output
        # -SjSi can be computed using an outer product between Sz and itself. Then
        # we add back Si for the i=j cases by adding a diagonal matrix with the
        # values of Si on its diagonal.
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D @ np.squeeze(grad0)
    
    # def backward(self, grad0, *args):
    #     # input s is softmax value of the original input x. Its shape is (1,n) 
    #     # i.e.  s = np.array([0.3,0.7]),  x = np.array([0,1])

    #     # make the matrix whose size is n^2.
    #     s = np.squeeze(self.output)
    #     tmp_ii = np.diag(- s * (1 - s)) # i = j
    #     tmp_ij = - self.output @ self.output.T # i != j
    #     tmp_ij -= np.diag(tmp_ij)
    #     jacobian = tmp_ii + tmp_ij
    #     input_gradient = jacobian @ np.squeeze(grad0)

    #     return input_gradient

