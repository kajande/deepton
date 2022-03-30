from random import seed
from random import random
from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

class Network:
    # Initialize a network
    def __init__(self, n_inputs=None, n_hidden=None, n_outputs=None, layers=None):
        # if layers:
        #     self.layers = layers
        # else:
        #     self.layers = self.from_layers(n_inputs, n_hidden, n_outputs)

        if not layers:
            layers = self.from_layers(n_inputs, n_hidden, n_outputs)
        self._layers = layers

    @staticmethod
    def from_layers(n_inputs, n_hidden, n_outputs):
        layers = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        layers.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        layers.append(output_layer)
        return layers

    @property
    def layers(self):
        return self._layers

    def __eq__(self, other):
        return self.layers == other.layers

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs