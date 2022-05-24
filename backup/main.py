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

filter_size = [3, 3, 8]
padding = [0, 0]
filter_stride = [1, 1]
pool_size = [2, 2]
pool_stride = [1, 1]

conv = convolutionalLayer('default', filter_size, padding, filter_stride) # 28x28x1 -> 26x26x8
pool = poolingLayer('Max', padding, pool_size, pool_stride) # 26x26x8 -> 13x13x8
out_p = pool.forward(conv.forward(x_train[0]))
sz = pool.output_size[0]*pool.output_size[1]*pool.output_size[2]
soft = fullyConnectedLayer('Softmax', sz, num_outputs) #13x13x8 -> 10


def cnn_forward(input, label): # 1 image by 1 image
    out_c = conv.forward(input)
    out_p = pool.forward(out_c)
    out_s = soft.forward(out_p)

    # Calculate Loss and Accuracy.
    loss = -np.log(out_s[label]) # Cross entropy
    acc = 1 if np.argmax(out_s) == label else 0 # If the highest probability is for the correct label, accuracy = 1, else 0.

    return out_c, out_p, out_s, loss, acc

def cnn_train(input, label, lr): # 1 image by 1 image
    # Full forward propagation
    _, __, out, loss, acc = cnn_forward(input, label)

    # Compute initial gradient
    grad0 = np.zeros(10) ## Number of classification categories
    grad0[label] = -1 / out[label]

    # Full back propagation
    grad_s = soft.backward(grad0, lr)
    grad_p = pool.backward(grad_s)
    grad_c = conv.backward(grad_p, lr)

    return grad_c, grad_p, grad_s, loss, acc

# _ = poolingLayer('Max', )
lr = 0.01
numEpochs = 10
for epoch in range(numEpochs):
    print('Running Epoch : %d' % (epoch+1))

    # Shuffle the training data
    shuffle_order = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_order]
    y_train = y_train[shuffle_order]

    # Training
    x = []
    y = []
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(x_train, y_train)): # Zip returns a tuple iterator, enumerate gives an iterator that counts them and returns their value at the same time.
        if i % 100 == 0: 
            print('%d steps out of 15 steps: Average Loss %.3f and Accuracy %d%%' % ((i/100 + 1), loss / 100, num_correct)) # There's 60k images in x_train.
            loss = 0
            num_correct = 0

        # Update loss and accuracy
        _, __, ___, dL, dA = cnn_train(im, label, lr)
        loss += dL
        num_correct += dA
        x.append((i+1)/100)
        y.append(loss) 

# Testing the CNN
print('**Testing phase')
loss = 0
num_correct = 0
for im, label in zip(x_test, y_test):
    _, __, ___, dL, dA = cnn_forward(im, label)
    loss += dL
    num_correct += dA

num_tests = len(x_test)
print('Test Loss: ', loss / num_tests)
print('Test Accuracy: ', num_correct / num_tests)

