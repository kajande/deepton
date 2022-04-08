class Neuron:
    def __init__(self, weights=None, output=None, error=None, delta=None):
        self.weights = weights
        self.output = output
        self.error = error
        self.delta = delta

    def __eq__(self, other):
        return self.weights == other.weights