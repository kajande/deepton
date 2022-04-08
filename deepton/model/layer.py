import random
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


class Layer:
    def __init__(self, n_inputs=None, n_outputs=None, neurons=None):
        if neurons is None:
            self.neurons = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_outputs)]
        else:
            self.neurons = neurons

    def __eq__(self, other) -> bool:
        return self.neurons == other.neurons

    def __iter__(self):
        return iter(self.neurons)

    def __len__(self):
        return len(self.neurons)

    def __getitem__(self, i):
        return self.neurons[i]

    def forward_propagate(self, inputs):
        new_inputs = []
        for neuron in self.neurons:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        return new_inputs

    def output_errors(self, expected, outputs):
        for j, neuron in enumerate(self.neurons):
            neuron['error'] = outputs[j] - expected[j]

    def backward_propagate_errors(self, next_layer):
        for j, neuron in enumerate(self.neurons):
            neuron['error'] = 0.0
            for next_neuron in next_layer:
                neuron['error'] += next_neuron['weights'][j] * next_neuron['delta']

    def backward_propagate_grads(self):
        for neuron in self.neurons:
            neuron['delta'] = neuron['error'] * transfer_derivative(neuron['output'])

    def update_weights(self, inputs, l_rate):
        for neuron in self.neurons:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']
