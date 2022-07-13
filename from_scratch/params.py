## IMPORTS.
import numpy as np
import matplotlib.pyplot as plt

from layers.convolutional import Convolutional
from layers.pooling import Pooling
from layers.dense import Dense
from layers.reshape import Reshape
from layers.softmax import Softmax
from utils.data_reader import get_data
from losses.losses import categorical_cross_entropy, categorical_cross_entropy_deriv
from model.network import *

## DATA FETCHING & PREPROCESSING.
dims = 1
num_train = 400
num_test = 100
(x_train, y_train), (x_test, y_test) = get_data('mnist', num_train, num_test, dims)
(x_train, y_train), (x_test, y_test) = preprocessing(x_train, x_test, y_train, y_test)

## MODEL PARAMETERS.
values = np.unique(y_test, return_counts = False)
num_outputs = len(values)

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
    Dense(num_outputs),
    Softmax()
)

## SHAPE AUTO-INFERRING.
infer_shape(network, x_train[0].shape)

## HYPER PARAMETERS.
lr = 0.005
num_epochs = 5

## TRAIN, TEST, EXPLANATION.
train(network, categorical_cross_entropy, categorical_cross_entropy_deriv, x_train, y_train,num_epochs, lr)
test(network, x_test, y_test)
saliency_map(network, categorical_cross_entropy, x_test[4], y_test[4])

## DEEPER LOOK.
kernels = network[0].kernels
num_kernels = kernels.shape[-1]
for n in range(num_kernels):
    kernel = kernels[..., n]
    plt.subplot(2, int(np.floor(num_kernels/2)) + 1, n + 1)
    plt.imshow(kernel, cmap = 'hot')
    plt.colorbar()

plt.title('trained convolution kernels')
plt.show()