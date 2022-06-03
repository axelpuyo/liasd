# DATASET
def get_data(str):
    if str == "mnist":
        from tensorflow.keras.datasets import mnist
        return mnist.load_data()
    else:
        pass
