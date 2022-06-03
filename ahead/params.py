import numpy as np
import matplotlib.pyplot as plt

from layers.convolutional_layer import convolutionalLayer
from layers.pooling_layer import poolingLayer
from layers.fully_connected_layer import fullyConnectedLayer
from layers.softmaxLayer import softmaxLayer
from utils.dataReader import get_data
from cgitb import grey

num = 1500
dims = 1
(x_train, y_train), (x_test, y_test) = get_data('mnist', num, dims)

filter_size = [3, 3, 8]
padding = [0, 0]
filter_stride = [1, 1]
pool_size = [2, 2]
pool_stride = [2, 2]
axis = (0,1)
