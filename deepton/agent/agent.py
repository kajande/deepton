from deepton.model.neuron import Neuron
from deepton.model.layer import Layer
from deepton.model.network import Network

class Agent:
    def __init__(self, metric):
        self.metric = metric

    # Evaluate an algorithm using a cross validation split
    def evaluate(self, algorithm, loader, *args, **kwargs):
        scores = list()
        for train_set, test_set, validation_set in loader:
            n_inputs = len(train_set[0]) - 1
            n_outputs = len(set([row[-1] for row in train_set]))
            network = Network(layers=[
                Layer(neurons=[Neuron(n_inputs) for _ in range(kwargs['n_hidden'])]),
                Layer(neurons=[Neuron(kwargs['n_hidden']) for _ in range(n_outputs)]),
            ])
            algorithm.train(network, train_set)
            predicted = network.test(test_set)
            actual = [row[-1] for row in validation_set]
            score = self.metric(actual, predicted)
            scores.append(score)
        return scores