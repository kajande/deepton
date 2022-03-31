from random import seed
from deepton.data.loader import CrossValidationSplitLoader

from deepton.model.network import Network

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

class Trainer:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch, n_hidden):
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.l_rate = l_rate

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def train(self, train_data, test):
        n_inputs = len(train_data[0]) - 1
        n_outputs = len(set([row[-1] for row in train_data]))
        # trainer = Trainer(l_rate, n_epoch, n_hidden, n_outputs, train, test)
        network = Network(n_inputs, n_outputs)
        network.init(self.n_hidden)
        network.learn(train_data, self)
        predictions = network.test(test)
        return predictions

    # Evaluate an algorithm using a cross validation split
    def evaluate(self, dataset, n_folds):
        cross_validation_split = CrossValidationSplitLoader(dataset, n_folds)
        scores = list()
        for train_set, test_set, validation_set in cross_validation_split:
            predicted = self.train(train_set, test_set)
            actual = [row[-1] for row in validation_set]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores