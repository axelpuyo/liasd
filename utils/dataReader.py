# DATASET
def get_data(str, num, dims):
    if str == "mnist":
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # put in kwargs
        x_train = x_train[:num]
        y_train = y_train[:num]
        x_test = x_test[:num]
        y_test = y_test[:num]
        
        # put in kwargs
        if dims != 0:
            import numpy as np
            x_train = np.repeat(x_train[..., np.newaxis], dims, axis = -1)
            x_test = np.repeat(x_train[..., np.newaxis], dims, axis = -1)
        return (x_train, y_train), (x_test, y_test)
    else:
        pass
