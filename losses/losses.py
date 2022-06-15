import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_deriv(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def categorical_cross_entropy(y_true, y_pred, num_classes):
    # print(y_true)
    # return - np.log(y_pred[y_true])
    return - 100*np.log(y_pred[int(y_true)] + 1e-15) / np.log(num_classes)

def categorical_cross_entropy_deriv(y_true, y_pred):
    grad0 = np.zeros(y_pred.shape)
    # grad0[int(y_true)] = - 1 / (y_pred[y_true] + 1e-15)
    grad0[int(y_true)] = - 1 / (y_pred[int(y_true)] + 1e-15)
    return grad0