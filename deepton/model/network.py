import random

from deepton.model.layer import Layer


class Network:
    # Initialize a network
    def __init__(self, layers=None):
        if layers is None:
            self._layers = []
        else:
            self._layers = layers

    @property
    def n_outputs(self):
        return self.layers[-1].n_outputs

    @property
    def n_inputs(self):
        return self.layers[0].n_inputs

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
    def forward_propagate(self, inputs):
        for layer in self._layers:
            layer.forward_propagate(inputs[:])
            inputs = layer.outputs

    def output_errors(self, expected):
        self.layers[-1].output_errors(expected)

    def output_grads(self):
        self.layers[-1].backward_propagate_grads()

    # Backpropagate error and store in neurons
    def backward_propagate_errors(self):
        for i in reversed(range(len(self._layers)-1)):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]
            layer.backward_propagate_errors(next_layer)

    def backward_propagate_grads(self):
        for layer in self.layers[:-1]:
            layer.backward_propagate_grads()

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i, layer in enumerate(self._layers):
            if i == 0:
                inputs = row[:-1]
            else:
                inputs = self._layers[i - 1].outputs
            layer.update_weights(inputs, l_rate)
    
    @property
    def outputs(self):
        return self.layers[-1].outputs

    # Make a prediction with a network
    def predict(self, row):
        self.forward_propagate(row)
        return self.outputs.index(max(self.outputs))


    def test(self, test):
        predictions = list()
        for row in test:
            prediction = self.predict(row)
            predictions.append(prediction)
        return(predictions)