# DATASET
def get_data(str):
    if str == "mnist":
        from keras.datasets import mnist
        return mnist.load_data()
    else:
        pass
