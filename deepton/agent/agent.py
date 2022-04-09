from deepton.model.neuron import Neuron
from deepton.model.layer import Layer
from deepton.model.network import Network

class Agent:
    # Evaluate an algorithm using a cross validation split
    def evaluate(self, algorithm, loader):
        scores = list()
        for train_set, test_set, validation_set in loader:
            n_inputs = len(train_set[0]) - 1
            n_outputs = len(set([row[-1] for row in train_set]))
            # network = Network(n_inputs, n_outputs)
            # network.init(n_hidden)
            network = Network(layers=[
                Layer(neurons=[Neuron(n_inputs) for _ in range(algorithm.n_hidden)]),
                Layer(neurons=[Neuron(algorithm.n_hidden) for _ in range(n_outputs)]),
            ])
            # print(f"\nnetwork.")
            # network.learn(train_set, self)
            algorithm.train(network, train_set)
            predicted = network.test(test_set)
            # predicted = algorithm.train(train_set, test_set, network)
            actual = [row[-1] for row in validation_set]
            score = algorithm.metric(actual, predicted)
            scores.append(score)
        return scores