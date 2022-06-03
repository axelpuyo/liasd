import numpy as np

def colors_load(n_train, n_test):
    ims = np.zeros((n_train + n_test, 32, 32, 3))
    labels = np.zeros((n_train + n_test, 1))

    for n in range(ims.shape[0]):
        idx = np.random.randint(3)
        ims[n, ..., idx] = np.random.randint(1,255)
        labels[n] = idx
 
    return (ims[:n_train], labels[:n_train]), (ims[-n_test:], labels[-n_test:])