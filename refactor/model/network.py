
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
        for i, (input, label) in enumerate(zip(x_train, y_train)):
            output = predict(network, input)
            # print('Predictions : ', *output.T, 'Labels : ', label)
            error += loss(label, output)
            if i % 10 == 0:
                if i == 0:
                    error = 0
                print(i/10, '/10 steps --- Average loss : ', error/10)
                error = 0

            grad = loss_deriv(label, output)
            # print(grad.shape)
            # print('Grad 0 : ', grad.T)
            for layer in reversed(network):
                # print(grad.shape)
                grad = layer.backward(grad, lr)
        
        error /= x_train.shape[0]
        print(f"{epoch + 1}/{num_epochs}, error={error}")

