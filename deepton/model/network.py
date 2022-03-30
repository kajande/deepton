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

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

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

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self._layers:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs


    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self))):
            layer = self[i]
            errors = list()
            if i != len(self)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self[i - 1]]
            for neuron in self[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= l_rate * neuron['delta']

    def learn(self, trainer):
        trainer.train(self)

    # Make a prediction with a network
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))