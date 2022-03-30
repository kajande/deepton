from random import seed
from random import random

class Network:
    # Initialize a network
    def __init__(self, n_inputs=None, n_hidden=None, n_outputs=None, layers=None):
        # if layers:
        #     self.layers = layers
        # else:
        #     self.layers = self.from_layers(n_inputs, n_hidden, n_outputs)

        if not layers:
            layers = self.from_layers(n_inputs, n_hidden, n_outputs)
        self.layers = layers

    @staticmethod
    def from_layers(n_inputs, n_hidden, n_outputs):
        layers = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        layers.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        layers.append(output_layer)
        return layers