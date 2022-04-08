from math import exp

class Neuron:
    def __init__(self, weights=None, output=None, error=None, delta=None):
        self.weights = weights
        self.output = output
        self.error = error
        self.delta = delta

    def __eq__(self, other):
        return self.weights == other.weights

    # Calculate neuron activation for an input
    def activate(self, inputs):
        activation = self.weights[-1]
        for i in range(len(self.weights)-1):
            activation += self.weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(self, activation):
        self.output = 1.0 / (1.0 + exp(-activation))

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)