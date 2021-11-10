import numpy as np
import random

class Network(object):

    # Constructor for a 3-layer network
    # @param: sizes is a list with each index referencing the number of
    #   neurons in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Feed forward method for network
    # Calculates the output of the network iteratively
    # @param: a is the input to the network
    # @pre-req: length of a is the same as the number of input neurons
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                n_test = len(test_data)
                print(f"Epoch {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")


# Sigmoid function for calculating a neuron's output
# @param: z is a vector/Numpy array
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))