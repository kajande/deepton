from random import seed
from deepton.data.loader import CrossValidationSplitLoader

from deepton.model.builder import Network

class Trainer:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch, seed=1):
        self.n_epoch = n_epoch
        self.l_rate = l_rate
        self.seed = seed

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def train(self, train_data, test, n_hidden):
        n_inputs = len(train_data[0]) - 1
        n_outputs = len(set([row[-1] for row in train_data]))
        network = Network(n_inputs, n_outputs)
        network.init(n_hidden, seed=1)
        network.learn(train_data, self)
        predictions = network.predictions(test)
        return predictions

    # Evaluate an algorithm using a cross validation split
    def evaluate(self, loader, metric, n_hidden):
        scores = list()
        for train_set, test_set, validation_set in loader:
            predicted = self.train(train_set, test_set, n_hidden)
            actual = [row[-1] for row in validation_set]
            score = metric(actual, predicted)
            scores.append(score)
        return scores