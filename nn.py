import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.w1, self.b1, self.w2, self.b2 = self.initialize_parameters(input_size)

    def initialize_parameters(self, n):
        """Initializes weights and biases randomly."""
        w1 = np.random.rand(10, n) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        w2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return w1, b1, w2, b2

    def relu(self, x):
        """Applies the ReLU activation function."""
        return np.maximum(x, 0)

    def softmax(self, x):
        """Applies the softmax activation function."""
        total = sum(np.exp(x))
        return np.exp(x) / total

    def forward_propagation(self, x):
        """Performs forward propagation through the network."""
        a0 = x
        z1 = self.w1.dot(a0) + self.b1
        a1 = self.relu(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)
        return z1, a1, z2, a2

    def one_hot_encode(self, Y):
        """One-hot encodes the target labels."""
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def backward_propagation(self, z1, a1, z2, a2, x, y):
        """Performs backward propagation to calculate gradients."""
        m = x.shape[1]
        y = self.one_hot_encode(y)

        dz2 = a2 - y
        dw2 = 1/m * (dz2.dot(a1.T))
        db2 = 1/m * (np.sum(dz2))

        dz1 = self.w2.T.dot(dz2) * (z1 > 0)
        dw1 = 1/m * (dz1.dot(x.T))
        db1 = 1/m * (np.sum(dz1))

        return dw1, db1, dw2, db2

    def update_parameters(self, dw1, db1, dw2, db2, alpha):
        """Updates the model's parameters using gradient descent."""
        self.w1 = self.w1 - alpha * dw1
        self.b1 = self.b1 - alpha * db1
        self.w2 = self.w2 - alpha * dw2
        self.b2 = self.b2 - alpha * db2

    def predict(self, a2):
        """Predicts class labels based on output probabilities."""
        return np.argmax(a2, 0)

    def accuracy(self, y_pred, y):
        """Calculates the accuracy of predictions."""
        return np.sum(y_pred == y) / y.size

    def train(self, x, y, alpha, epochs):
        """Trains the neural network on provided data."""
        for i in range(epochs):
            z1, a1, z2, a2 = self.forward_propagation(x)
            dw1, db1, dw2, db2 = self.backward_propagation(z1, a1, z2, a2, x, y)
            self.update_parameters(dw1, db1, dw2, db2, alpha)

            if (i % 20) == 0:
                print("Iteration number: ", i)
                print("Accuracy = ", self.accuracy(self.predict(a2), y))


# Example usage (assuming data loading and preprocessing is done as in original script):
# ... (data loading and preprocessing code)


data=pd.read_csv('numbers.csv')
data=np.array(data)
np.random.shuffle(data)
m=data.shape[0]


#24X24 size images
train=data[:int(m*0.8), :] #80% train data
test=data[int(m*0.8):m,:] #20% test data

x_train=train[:,1:].T
y_train=train[:,0]
x_test=test[:,1:].T
y_test=test[:,0]

x_train=x_train/255
x_test=x_test/255

nn = NeuralNetwork(x_train.shape[0])
nn.train(x_train, y_train.flatten(), 0.1, 200)  # Note: y_train is flattened here

