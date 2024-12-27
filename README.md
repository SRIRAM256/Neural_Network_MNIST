initialize_parameters(n): Initializes the weights and biases randomly based on the input size (n).

relu(x): Implements the Rectified Linear Unit (ReLU) activation function.

softmax(x): Implements the softmax activation function for output layer normalization.

forward_propagation(x): Performs forward propagation through the network, calculating intermediate activations and outputs.
one_hot_encode(Y): Converts target labels into one-hot encoded format.
backward_propagation(z1, a1, z2, a2, x, y): Calculates gradients of the loss function with respect to the network's parameters.
update_parameters(dw1, db1, dw2, db2, alpha): Updates the network's weights and biases using gradient descent.
predict(a2): Predicts class labels based on the output probabilities.
accuracy(y_pred, y): Computes the accuracy of the predictions.
train(x, y, alpha, epochs): Trains the network using the provided training data, learning rate (alpha), and number of epochs. Prints the training accuracy at regular intervals.
