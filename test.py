from layers.model import *
from params import *

model = model()
model.preprocessing(x_train, y_train)
model.add_layer(convolutionalLayer, 'default', filter_size, padding, filter_stride)
model.add_layer(poolingLayer, 'Max', padding, pool_size, pool_stride, axis)
model.add_layer(convolutionalLayer, 'default', filter_size, padding, filter_stride)
model.add_layer(poolingLayer, 'Max', padding, pool_size, pool_stride, axis)
model.add_layer(fullyConnectedLayer, 'Softmax', 10)

f_out, loss, acc = model.forward(x_train[0], y_train[0])

grad0 = np.zeros(10)
print(y_train[0])
grad0[y_train[0]] = -1 / f_out[y_train[0]]

b_out = model.backward(grad0, 0.01)

model.fit(x_train, y_train, 5, 0.01)
