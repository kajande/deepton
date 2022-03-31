from random import seed

from deepton.model.network import Network


class Trainer:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch, n_outputs, train, test=None):
        self._train = train
        self._test = test
        self._n_epoch = n_epoch
        self._n_outputs = n_outputs
        self._l_rate = l_rate

    def train(self, network):
        for epoch in range(self._n_epoch):
            for row in self._train:
                outputs = network.forward_propagate(row)
                expected = [0 for i in range(self._n_outputs)]
                expected[row[-1]] = 1
                network.backward_propagate_error(expected)
                network.update_weights(row, self._l_rate)