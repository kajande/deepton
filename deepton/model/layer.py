import random

from deepton.model.neuron import Neuron

def transfer_derivative(output):
    return output * (1.0 - output)

class Layer:
    def __init__(self, neurons=None):
        if neurons is None:
            self.neurons = []
        else:
            self.neurons = neurons

    @property
    def n_inputs(self):
        return len(self.neurons[0].weights)-1

    @property
    def n_outputs(self):
        return len(self.neurons)

    def __eq__(self, other) -> bool:
        return self.neurons == other.neurons

    def __iter__(self):
        return iter(self.neurons)

    def __len__(self):
        return len(self.neurons)

    def __getitem__(self, i):
        return self.neurons[i]

    def forward_propagate(self, inputs):
        for neuron in self.neurons:
            neuron.forward_propagate(inputs)

    @property
    def outputs(self):
        return [neuron.output for neuron in self.neurons]

    def output_errors(self, expected):
        for j, neuron in enumerate(self.neurons):
            neuron.output_errors(expected[j])

    def backward_propagate_errors(self, next_layer):
        for j, neuron in enumerate(self.neurons):
            neuron.error = 0.0
            for next_neuron in next_layer:
                neuron.error += next_neuron.weights[j] * next_neuron.delta

    def backward_propagate_grads(self):
        for neuron in self.neurons:
            neuron.backward_propagate_grads()
            
    def update_weights(self, inputs, l_rate):
        for neuron in self.neurons:
            for j in range(len(inputs)):
                neuron.weights[j] -= l_rate * neuron.delta * inputs[j]
            neuron.weights[-1] -= l_rate * neuron.delta
