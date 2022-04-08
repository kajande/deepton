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
    def forward_propagate(self, inputs):
        activation = self.weights[-1]
        for i in range(len(self.weights)-1):
            activation += self.weights[i] * inputs[i]
        self.activation = activation

    # Transfer neuron activation
    def activate(self):
        self.output = 1.0 / (1.0 + exp(-self.activation))

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)