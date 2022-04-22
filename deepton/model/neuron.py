from math import exp
import random

class Neuron:
    def __init__(self, n_inputs=None, weights=None, output=None, error=None, delta=None):
        if n_inputs:
            self.weights = [random.random() for i in range(n_inputs + 1)]
        else:
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
        self.activate()

    # Transfer neuron activation
    def activate(self):
        self.output = 1.0 / (1.0 + exp(-self.activation))

    def output_errors(self, expected):
        self.error = self.output - expected

    # Calculate the derivative of an neuron output
    def backward_propagate_grads(self):
        self.delta = self.error * (self.output * (1.0 - self.output))