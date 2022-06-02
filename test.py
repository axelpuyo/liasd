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

# outputs = model.forward(x_train[0], y_train[0])

# TEST TO SEE IF CONVOLUTION FORWARD ISNT BROKEN
conv = convolutionalLayer('default', filter_size, padding, filter_stride)
out_c = conv.forward(x_train[4])

plt.subplot(1,2,1)
plt.imshow(x_train[4,...,0], cmap = 'Reds')
plt.colorbar()
# plt.subplot(1,4,2)
# plt.imshow(x_train[4,...,1], cmap = 'Greens')
# plt.colorbar()
# plt.subplot(1,4,3)
# plt.imshow(x_train[4,...,2], cmap = 'Blues')
# plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(out_c[...,0])
plt.colorbar()
plt.show()

# predictions = outputs[-1]
# grad0 = np.zeros(10)
# grad0[int(y_train[0])] = -1 / predictions[int(y_train[0])]

# gradients = model.backward(grad0, 0.01)
# grad_image = gradients[-1]
# plt.imshow(grad_image[..., 0], cmap = 'gray')
# plt.colorbar()
# plt.show()

model.fit(x_train, y_train, 15, 0.001)
