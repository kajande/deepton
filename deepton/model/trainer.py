from random import seed
from deepton.data.loader import CrossValidationSplitLoader

from deepton.model.network import Network

class Trainer:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch, seed=1):
        self.n_epoch = n_epoch
        self.l_rate = l_rate
        self.seed = seed

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def train(self, network, train_data):
        # trainer.train(self)
        # self.init(trainer.n_hidden, trainer.seed) # later refactor to `trainer.initializer` instead
        for epoch in range(self.n_epoch):
            for row in train_data:
                outputs = network.forward_propagate(row)
                expected = [0 for i in range(network.n_outputs)]
                # print(f"\n\nn_outputs: {trainer.n_outputs}\n\n")
                expected[row[-1]] = 1
                network.output_errors(expected, outputs)
                network.output_grads()
                network.backward_propagate_errors()
                network.backward_propagate_grads()
                network.update_weights(row, self.l_rate)
        # update here the `trainer.initializer` parameters


    # Evaluate an algorithm using a cross validation split
    def evaluate(self, loader, metric, n_hidden):
        scores = list()
        for train_set, test_set, validation_set in loader:
            n_inputs = len(train_set[0]) - 1
            n_outputs = len(set([row[-1] for row in train_set]))
            network = Network(n_inputs, n_outputs)
            network.init(n_hidden, seed=1)
            # network.learn(train_set, self)
            self.train(network, train_set)
            predicted = network.predictions(test_set)
            # predicted = self.train(train_set, test_set, network)
            actual = [row[-1] for row in validation_set]
            score = metric(actual, predicted)
            scores.append(score)
        return scores