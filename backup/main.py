import numpy as np
import matplotlib.pyplot as plt

from layers.convolutional_layer import convolutionalLayer
from layers.pooling_layer import poolingLayer
from layers.fully_connected_layer import fullyConnectedLayer
from utils.dataReader import get_data
from cgitb import grey

(x_train, y_train), (x_test, y_test) = get_data("mnist")
x_train = x_train / 255 
x_test = x_test / 255

x_train = x_train[:1500]
x_test = x_test[:1500]
y_train = y_train[:1500]
y_test = y_test[:1500]

num_outputs = len(np.unique(y_test))
input_height = x_train[0].shape[0]
input_width = x_train[0].shape[1]
# print(num_outputs, height, width)

filter_size = [3, 3, 8]
padding = [0, 0]
filter_stride = [1, 1]
pool_size = [2, 2]
pool_stride = [2, 2]
new_height = (input_height - filter_size[0] + 1)//pool_size[0]
new_width = (input_width - filter_size[1] + 1)//pool_size[1]
# print(new_height, new_width)

conv = convolutionalLayer('default', filter_size, padding, filter_stride) # 28x28x1 -> 26x26x8
pool = poolingLayer('Max', pool_size, pool_stride) # 26x26x8 -> 13x13x8
soft = fullyConnectedLayer('Softmax', new_height*new_width*8, num_outputs) #13x13x8 -> 10


def cnn_forward_prop(input, label):
    out_c = conv.forward(input)
    out_p = pool.forward(out_c)
    out_s = soft.forward(out_p)

    # Calculate Loss and Accuracy.
    loss = -np.log(out_s[label]) # Cross entropy
    acc = 1 if np.argmax(out_s) == label else 0 # If the highest probability is for the correct label, accuracy = 1, else 0.

    return out_c, out_p, out_s, loss, acc

    
out_c, out_p, out_s, loss, acc = cnn_forward_prop(x_train[0], y_train[0])

grad0 = np.zeros(10)
grad0[5] = -1/out_s[5]

grads = soft.backward(grad0, 0.01)
gradp = pool.backward(grads)
gradc= conv.backward(gradp, 0.01)

plt.subplot(1,3,1)
plt.imshow(gradc[:,:,0], cmap = 'gray')
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(gradp[:,:,0], cmap = 'gray')
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(grads[:,:,0], cmap = 'gray')
plt.colorbar()

plt.show()
##

def training_cnn(input, label, lr):
    # Full forward propagation
    _, __, out, loss, acc = cnn_forward_prop(input, label)

    # Compute initial gradient
    grad = np.zeros(10) ## Number of classification categories
    grad[label] = -1 / out[label]

    # Full back propagation
    grad_back = soft.backward(grad, lr)
    grad_back = pool.backward(grad_back)
    grad_back = conv.backward(grad_back, lr)

    return loss, acc

lr = 0.01
numEpochs = 10
for epoch in range(numEpochs):
    print('Running Epoch : %d' % (epoch+1))

    # Shuffle the training data
    shuffle_order = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_order]
    y_train = y_train[shuffle_order]

    # Training
    loss = 0
    num_correct = 0
    x = []
    y = []
    for i, (im, label) in enumerate(zip(x_train, y_train)): # Zip returns a tuple iterator, enumerate gives an iterator that counts them and returns their value at the same time.
        if i % 100 == 0: 
            print('%d steps out of 15 steps: Average Loss %.3f and Accuracy %d%%' % ((i/100 + 1), loss / 100, num_correct)) # There's 60k images in x_train.
            loss = 0
            num_correct = 0

        # Update loss and accuracy
        dL, dA = training_cnn(im, label, lr)
        loss += dL
        num_correct += dA
        x.append((i+1)/100)
        y.append(loss) 

# Testing the CNN
print('**Testing phase')
loss = 0
num_correct = 0
for im, label in zip(x_test, y_test):
    _, __, ___, dL, dA = cnn_forward_prop(im, label)
    loss += dL
    num_correct += dA

num_tests = len(x_test)
print('Test Loss: ', loss / num_tests)
print('Test Accuracy: ', num_correct / num_tests)



