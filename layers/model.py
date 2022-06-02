import numpy as np

from layers.convolutional_layer import convolutionalLayer
from layers.pooling_layer import poolingLayer
from layers.fully_connected_layer import fullyConnectedLayer

class model:
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.last_layer_num = -1
    
    def preprocessing(self, x_train, x_test, y_train, y_test):
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
        self.num_classes = len(values)
        print('----training----')
        print('labels : counts')
        for i in range(len(values)):
            print(' ', values[i], '   : ', counts[i])

        values, counts = np.unique(y_test, return_counts = True)
        self.num_classes = len(values)
        print('----testing----')
        print('labels : counts')
        for i in range(len(values)):
            print(' ', values[i], '   : ', counts[i])
        return (x_train, y_train), (x_test, y_test)

    def add_layer(self, layer_type, *args):
        self.layers.append(layer_type(*args))
        self.num_layers += 1
        self.last_layer_num += 1
    
    def forward(self, input, label):
        predictions = []
        for n in range(len(self.layers)):
            output = self.layers[n].forward(input)
            predictions.append(output)
            input = output

        return predictions
    
    # def initialize_weights(self, input, label):
    #     predictions = []
    #     for n in range(len(self.layers)):
    #         self.layers[n].initialize_weights()
    #         output = self.layers[n].forward(input)
    #         predictions.append(output)
    #         if n != len(self.layers)-1:
    #             flat = output.flatten()
    #             self.layers[n+1].numInputs = flat.shape[0]
    #         input = output

    def backward(self, input, lr):
        gradients = []
        for n in range(len(self.layers)):
            gradient = self.layers[-(n+1)].backward(input, lr)
            gradients.append(gradient)
            input = gradient

        return gradients

    def train(self, input, label, lr): # 1 image by 1 image
        # Full forward propagation
        predictions = self.forward(input, label)
        results = predictions[-1]

        loss = -np.log(results[int(label)]) # Categorical cross entropy
        accuracy = 1 if np.argmax(results) == label else 0 # If the highest probability is for the correct label, accuracy = 1, else 0.
        # print(results)
        # Compute initial gradient
        grad0 = np.zeros(self.num_classes) ## Number of classification categories
        grad0[int(label)] = -1 / results[int(label)]

        # Full back propagation
        gradients = self.backward(grad0, lr)

        return predictions, gradients, loss, accuracy

    def fit(self, inputs, labels, num_epochs, lr):
        for epoch in range(num_epochs):
            print('Running Epoch : %d' % (epoch+1))

            # Shuffle the training data
            # shuffle_order = np.random.permutation(len(inputs))
            # inputs = inputs[shuffle_order]
            # labels = labels[shuffle_order]

            # Training
            x = []
            y = []
            loss = 0
            num_correct = 0
            for i, (im, label) in enumerate(zip(inputs, labels)): # Zip returns a tuple iterator, enumerate gives an iterator that counts them and returns their value at the same time.
                if i % 100 == 0: 
                    print('%d steps out of 15 steps: Average Loss %.3f and Accuracy %d%%' % ((i/100 + 1), loss / 100, num_correct)) # There's 60k images in x_train.
                    loss = 0
                    num_correct = 0

                # Update loss and accuracy
                predictions, gradients, dL, dA = self.train(im, label, lr)
                
                bool = gradients[-1].any()
                if i == 0:
                    print(bool)
                else:
                    if not bool:
                        print(bool)
                
                loss += dL
                num_correct += dA
                x.append((i+1)/100)
                y.append(loss) 
