import numpy as np
import matplotlib.pyplot as plt

from numba import jit, cuda
from layers.convolutional import Convolutional
from layers.pooling import Pooling
from layers.dense import Dense
from layers.reshape import Reshape
from layers.softmax import Softmax
from utils.dataReader import get_data
from losses.losses import categorical_cross_entropy, categorical_cross_entropy_deriv
from model.network import *

num = 200
dims = 1
(x_train, y_train), (x_test, y_test) = get_data('cifar10', num, dims)
num_outputs, (x_train, y_train), (x_test, y_test) = preprocessing(x_train, x_test, y_train, y_test)
shape = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]

filter_size = [3, 3, 15]
filter_padding = [0, 0]
filter_stride = [1, 1]

pool_size = [2, 2]
pool_padding = [0, 0]
pool_stride = [2, 2]

network = (
    Convolutional('default', filter_size, filter_padding, filter_stride),
    Pooling(pool_size, pool_padding, pool_stride),
    Reshape(),
    Dense(3375, num_outputs), # need to automate dense layer input shape acquisition.
    # Softmax(),
    # Dense(100, num_outputs),
    Softmax()
)


num_epochs = 5
lr = 0.001
train(network, categorical_cross_entropy, categorical_cross_entropy_deriv, x_train, y_train,num_epochs, lr)
# test(network, x_test, y_test)
saliency_map(network, categorical_cross_entropy, x_train[4], y_train[4])