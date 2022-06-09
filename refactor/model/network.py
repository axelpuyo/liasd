import numpy as np

def preprocessing(x_train, x_test, y_train, y_test):
    # if x_train.ndim < 4:
     #     x_train = x_train[..., np.newaxis]
    #     x_test = x_test[..., np.newaxis]
    # if x_train.ndim > 4:
    #     x_train = np.squeeze(x_train)
    #     x_test = np.squeeze(x_test)

    print('Normalizing samples')
    x_train = x_train / 255
    x_test = x_test / 255

    values, counts = np.unique(y_train, return_counts = True)
    print('----training----')
    print('labels : counts')
    for i in range(len(values)):
        print(' ', values[i], '   :  ', counts[i])

    values, counts = np.unique(y_test, return_counts = True)
    num_classes = len(values)
    print('----testing----')
    print('labels : counts')
    for i in range(len(values)):
        print(' ', values[i], '   :  ', counts[i])
    return num_classes, (x_train, y_train), (x_test, y_test)

def predict(network, input):
    outputs = []
    for layer in network:
        output = layer.forward(input)
        outputs.append(output)
        input = output
    return output

def train(network, loss, loss_deriv, x_train, y_train, num_epochs, lr):
    for epoch in range(num_epochs):
        error = 0
        shuffle_order = np.random.permutation(len(x_train))
        x_train = x_train[shuffle_order]
        y_train = y_train[shuffle_order]

        for i, (input, label) in enumerate(zip(x_train, y_train)):
            output = predict(network, input)
            error += loss(label, output)
            if i % (x_train.shape[0] / 10) == 0:
                if i == 0:
                    error = 0
                print(i / (x_train.shape[0] / 10), '/ 10 steps -- average loss : ', float(error / (x_train.shape[0] / 10)))
                print('pred : ', np.argmax(output), 'true : ', int(label))
                print('p_val : ', round(float(output[np.argmax(output)]), 3) , 't_val : ', round(float(output[int(label)]), 3))
                error = 0

            grad = loss_deriv(label, output)
            # print(grad.shape)
            # print('Grad 0 : ', grad.T)
            for layer in reversed(network):
                # print(grad.shape)
                grad = layer.backward(grad, lr)
        
        error /= x_train.shape[0]
        print(f"{epoch + 1}/{num_epochs}, error={float(error)}")

def test(network, x_test, y_test):
    counter = 0
    for i, (x,y) in enumerate(zip(x_test, y_test)):
        output = predict(network, x)
        print('pred : ', np.argmax(output), 'true : ', int(y))
        if np.argmax(output) == int(y):
            counter += 1

    print('acc : ', counter/i)

def confusion_matrix():
    pass
