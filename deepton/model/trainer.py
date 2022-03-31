from random import seed

from deepton.model.network import Network


class Trainer:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch, n_outputs, train, test=None):
        self.train_data = train
        self.test_data = test
        self.n_epoch = n_epoch
        self.n_outputs = n_outputs
        self.l_rate = l_rate

    def train(self, network):
        pass

    def evaluate(self, model, loader):
        pass