# Backprop on the Seeds Dataset
from deepton.data.loader import CrossValidationSplitLoader
from deepton.model.network import Network
from deepton.model.trainer import Trainer

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	cross_validation_split = CrossValidationSplitLoader(dataset, n_folds)
	scores = list()
	for train_set, test_set, validation_set in cross_validation_split:
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in validation_set]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	trainer = Trainer(l_rate, n_epoch, n_outputs, train, test)
	network = Network(n_inputs, n_hidden, n_outputs)
	network.learn(trainer)
	predictions = network.test(test)
	return predictions
