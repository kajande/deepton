from random import seed
from deepton.data.loader import CrossValidationSplitLoader

from deepton.model.builder import Network

class Trainer:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch, seed=1):
        self.n_epoch = n_epoch
        self.l_rate = l_rate
        self.seed = seed

    # Evaluate an algorithm using a cross validation split
    def evaluate(self, loader, metric, n_hidden):
        scores = list()
        for train_set, test_set, validation_set in loader:
            n_inputs = len(train_set[0]) - 1
            n_outputs = len(set([row[-1] for row in train_set]))
            network = Network(n_inputs, n_outputs)
            network.init(n_hidden, seed=1)
            network.learn(train_set, self)
            predicted = network.predictions(test_set)
            # predicted = self.train(train_set, test_set, network)
            actual = [row[-1] for row in validation_set]
            score = metric(actual, predicted)
            scores.append(score)
        return scores