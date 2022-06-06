
def predict(network, input):
    outputs = []
    for layer in network:
        output = layer.forward(input)
        outputs.append(output)
        input = output
    return output

def train(network, loss, loss_deriv, x_train, y_train, num_epochs, lr):
    for epoch in num_epochs:
        error = 0
        for input, label in zip(x_train, y_train):
            output = predict(network, input)

            error += loss(label, output)

            grad = loss_deriv(label, output)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)
        
        error /= len(x_train[0])
        print(f"{epoch + 1}/{num_epochs}, error={error}")

