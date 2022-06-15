import numpy as np
import matplotlib.pyplot as plt

# @cuda.jit
def preprocessing(x_train, x_test, y_train, y_test):
    if x_train.ndim < 4:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    if x_train.ndim > 4:
        x_train = np.squeeze(x_train)
        x_test = np.squeeze(x_test)

    print('normalizing samples')
    x_train = x_train / 255
    x_test = x_test / 255

    values, counts = np.unique(y_train, return_counts = True)
    print('----training----')
    print('labels : counts')
    for i in range(len(values)):
        print(' ', values[i], '   :  ', counts[i])

    values, counts = np.unique(y_test, return_counts = True)
    print('----testing----')
    print('labels : counts')
    for i in range(len(values)):
        print(' ', values[i], '   :  ', counts[i])
    return (x_train, y_train), (x_test, y_test)

def infer_shape(network, input_shape):
    print('---inferring shapes---')
    for layer in network:
        input_shape = layer.infer_shape(input_shape)

def predict(network, input):
    outputs = []
    for layer in network:
        output = layer.forward(input)
        outputs.append(output)
        input = output
    return output

def train(network, loss, loss_deriv, x_train, y_train, num_epochs, lr):
    values = np.unique(y_train, return_counts = False)
    num_outputs = len(values)
    for epoch in range(num_epochs):
        error = 0
        shuffle_order = np.random.permutation(len(x_train))
        x_train = x_train[shuffle_order]
        y_train = y_train[shuffle_order]

        for i, (input, label) in enumerate(zip(x_train, y_train)):
            output = predict(network, input)
            error += loss(label, output, num_outputs)
            if i % (x_train.shape[0] / 10) == 0:
                if i == 0:
                    error = 0
                print('# ', i / (x_train.shape[0] / 10), '/ 10th of dataset seen - batch average loss : ', np.round(float(error / (x_train.shape[0] / 10)), 1), '% #')
                print('pred : ', np.argmax(output), 'true : ', int(label))
                print('p_val : ', round(float(output[np.argmax(output)]), 3) , 't_val : ', round(float(output[int(label)]), 3))
                error = 0

            grad = loss_deriv(label, output)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)
        
        error /= x_train.shape[0]
        print(f"## running epoch {epoch + 1}/{num_epochs}, accuracy = {np.round(float(error), 1)}%")

def test(network, x_test, y_test):
    counter = 0
    for i, (x,y) in enumerate(zip(x_test, y_test)):
        output = predict(network, x)
        print('pred : ', np.argmax(output), 'true : ', int(y))
        if np.argmax(output) == int(y):
            counter += 1

    print('acc : ', np.round(100*counter/i, 1), '%')

def confusion_matrix():
    pass

def saliency_map(network, loss, input, label): # Takes SUPER long. Implement RISE instead ?
    from utils.label_names import get_string

    plt.subplot(1,2,1)
    plt.imshow(input[..., 0], cmap = 'gray')
    plt.title(get_string('cifar10', label))
    plt.colorbar()

    output = predict(network, input)
    best_loss = loss(label, output, 10) # 10 nombre de classes
    map = np.zeros(input.shape)
    print('-- computing saliency map ---')
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            new_input = input - input[i, j]
            new_output = predict(network, new_input)
            new_loss = loss(label, new_output, 10) # ce 3 c'est num_outputs
            map[i, j] = best_loss - new_loss

    plt.subplot(1,2,2)
    plt.imshow(map, cmap = 'hot')
    plt.title('saliency map (percent of contribution)')
    plt.colorbar()
    plt.show()



