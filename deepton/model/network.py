from random import seed
from random import random

class Network:
    # Initialize a network
    def __init__(self, n_inputs, n_hidden, n_outputs):
        if not n_inputs or not n_hidden or not n_outputs:
            self.layers = None
        else:
            self.layers = list()
            hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            self.layers.append(hidden_layer)
            output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
            self.layers.append(output_layer)
    
    @classmethod
    def from_layers(cls, layers):
        self = cls(None, None, None)
        self.layers = layers
        return self