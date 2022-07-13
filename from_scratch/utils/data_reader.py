## DATASET
import numpy as np

def get_data(str, num_train, num_test, dims):
    if str == 'cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = np.squeeze(x_train[:num_train])
        y_train = y_train[:num_train]
        x_test = np.squeeze(x_test[:num_test])
        y_test = y_test[:num_test]

        if dims != 0:
            x_train = np.repeat(x_train[..., np.newaxis], dims, axis = -1)
            x_test = np.repeat(x_test[..., np.newaxis], dims, axis = -1)  

    if str == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # put in kwargs
        x_train = x_train[:num_train]
        y_train = y_train[:num_train]
        x_test = x_test[:num_test]
        y_test = y_test[:num_test]
        
        # put in kwargs
        if dims != 0:
            x_train = np.repeat(x_train[..., np.newaxis], dims, axis = -1)
            x_test = np.repeat(x_test[..., np.newaxis], dims, axis = -1)

    elif str == 'colors':
        from utils.dataset_creator import colors_load
        (x_train, y_train), (x_test, y_test) = colors_load(num_train, num_test)
    
    return (x_train, y_train), (x_test, y_test)