from layers.model import *
from params import *

model = model()
(x_train, y_train), (x_test, y_test) = model.preprocessing(x_train, x_test, y_train, y_test)
model.add_layer(convolutionalLayer, 'default', filter_size, padding, filter_stride)
model.add_layer(poolingLayer, 'Max', padding, pool_size, pool_stride, axis) ; 
output = model.forward(x_train[0], y_train[0])
num_inputs = output[-1].flatten().shape[0]
model.add_layer(fullyConnectedLayer, 'Softmax', num_inputs, 10)
model.add_layer(softmaxLayer, 10)

model.fit(x_train, y_train, 15, 0.1)
