import numpy as np

from layers.convolutional_layer import convolutionalLayer
from layers.pooling_layer import poolingLayer
from layers.fully_connected_layer import fullyConnectedLayer

class model:
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.last_layer_num = -1
    
    def preprocessing(self, inputs, labels):
        values, counts = np.unique(labels, return_counts = True)
        self.final_outputs = len(values)
        print('labels : counts')
        for i in range(len(values)):
            print(' ', values[i], '   : ', counts[i])
        

    def add_layer(self, type, *args):
        self.layers.append(type(*args))
        self.num_layers += 1
        self.last_layer_num += 1
    
    def forward(self, input, label):
        for n in range(len(self.layers)):
            output = self.layers[n].forward(input)
            input = output

        loss = -np.log(output[int(label)]) # Cross entropy
        acc = 1 if np.argmax(input) == label else 0 # If the highest probability is for the correct label, accuracy = 1, else 0.
        
        return output, loss, acc
    
    def backward(self, input, lr):
        for n in range(len(self.layers)):
            print(self.layers[-(n+1)])
            output = self.layers[-(n+1)].backward(input, lr)
            input = output
        return output

    def train(self, input, label, lr): # 1 image by 1 image
        # Full forward propagation
        out, loss, acc = self.forward(input, label)

        # Compute initial gradient
        grad0 = np.zeros(self.final_outputs) ## Number of classification categories
        grad0[int(label)] = -1 / out[int(label)]

        # Full back propagation
        out_b = self.backward(input, lr)

        return out, out_b, loss, acc

    def fit(self, input, label, num_epochs, lr):
        for epoch in range(num_epochs):
            print('Running Epoch : %d' % (epoch+1))

            # Shuffle the training data
            shuffle_order = np.random.permutation(len(input))
            input = input[shuffle_order]
            input = input[shuffle_order]

            # Training
            x = []
            y = []
            loss = 0
            num_correct = 0
            for i, (im, label) in enumerate(zip(input, label)): # Zip returns a tuple iterator, enumerate gives an iterator that counts them and returns their value at the same time.
                if i % 100 == 0: 
                    print('%d steps out of 15 steps: Average Loss %.3f and Accuracy %d%%' % ((i/100 + 1), loss / 100, num_correct)) # There's 60k images in x_train.
                    loss = 0
                    num_correct = 0

                # Update loss and accuracy
                _, __, ___, dL, dA = self.train(im, label, lr)
                loss += dL
                num_correct += dA
                x.append((i+1)/100)
                y.append(loss) 
