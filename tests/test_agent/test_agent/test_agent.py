import unittest

from deepton.data.loader import CrossValidationSplitLoader
from deepton.data.transformer import NormalizeTransform
from deepton.data.analysis import minmax
from deepton.training.metrics import accuracy
from deepton.agent.agent import Agent
from deepton.training.algorithm import Backpropagation

class TestAgent(unittest.TestCase):
    pass

def TestEvaluate(TestAgent):
    def test_basic(self):
        # Test Backprop on Seeds dataset
        # print("Testing back_propagation algorithm:")
        # normalize input variables
        normalize = NormalizeTransform()
        normalize.fit(self.extracted.data, minmax)
        normalized = normalize(self.extracted.data)
        n_folds = 5
        loader = CrossValidationSplitLoader(normalized, n_folds)
        # evaluate algorithm
        l_rate = 0.3
        n_epoch = 500
        n_hidden = 5
        algorithm = Backpropagation(l_rate, n_epoch, n_hidden, accuracy)
        agent = Agent()
        scores = agent.evaluate(algorithm, loader)
        # scores = evaluate_algorithm(self.dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        # print('Scores: %s' % scores)
        self.assertListEqual(scores, [100.0, 100.0, 100.0, 100.0, 100.0])
        mean_accuracy = sum(scores)/float(len(scores))
        # print('Mean Accuracy: %.3f%%' % (mean_accuracy))
        self.assertEqual(mean_accuracy, 100.000)