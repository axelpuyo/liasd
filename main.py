## IMPORTS
from params import *

## PREPROCESSING
values, counts = np.unique(y_train, return_counts = True)
print('labels : counts')
for i in range(len(values)):
    print(' ', values[i], '   : ', counts[i])

## 

conv = convolutionalLayer('default', filter_size, padding, filter_stride) # 28x28x1 -> 26x26x8
pool = poolingLayer('Max', padding, pool_size, pool_stride, axis) ; out_c = conv.forward(x_train[0]) ; out_p = pool.forward(out_c) ; sz = pool.output_size[0]*pool.output_size[1]*pool.output_size[2]
soft = fullyConnectedLayer('Softmax', sz, len(np.unique(values)))

out_c = conv.forward(x_train[0])
out_p = pool.forward(out_c)
out_s = soft.forward(out_p)

loss = -np.log(out_s[y_train[0]])

grad0 = np.zeros(10)
grad0[y_train[0]] = - 1/out_s[y_train[0]]

grad = soft.backward(grad0, 0.01)
grad = pool.backward(grad)
grad = conv.backward(grad, 0.01)
print(np.all(grad==0))

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

