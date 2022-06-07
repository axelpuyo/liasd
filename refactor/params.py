import numpy as np
import matplotlib.pyplot as plt

from layers.convolutional import Convolutional
from layers.pooling import Pooling
from layers.dense import Dense
from layers.reshape import Reshape
from layers.softmax import Softmax
from activations.activations import ReLu, Sigmoid
from utils.convolutions import convolution
from utils.dataReader import get_data
from losses.losses import categorical_cross_entropy, categorical_cross_entropy_deriv
from model.network import *
from cgitb import grey

num = 100
dims = 1
(x_train, y_train), (x_test, y_test) = get_data('mnist', num, dims)

x_train = x_train / 255
x_test = x_test / 255

filter_size = [3, 3, 5]
filter_padding = [0, 0]
filter_stride = [1, 1]

pool_size = [2, 2]
pool_padding = [0, 0]
pool_stride = [2, 2]

# axis = (0,1)

network = (
    Convolutional('default', filter_size, filter_padding, filter_stride),
    Pooling(pool_size, pool_padding, pool_stride),
    Reshape(),
    Dense(13*13*5, 100),
    Dense(100, 10),
    Softmax()
    # Softmax()
)

out = predict(network, x_train[2])
# print(out)
# print(out.shape, np.diag(np.array(out)))

train(network, categorical_cross_entropy, categorical_cross_entropy_deriv, x_train, y_train, 5, 0.001)
