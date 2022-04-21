import random
import unittest
from deepton.data.extractor import Extract

from deepton.data.loader import CrossValidationSplitLoader
from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.data.analysis import minmax
from deepton.agent.metrics import accuracy
from deepton.agent.agent import Agent
from deepton.training.algorithm import Backpropagation

class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1)
        # load and prepare data
        filename = 'example.csv'
        # filename = 'seeds_dataset.csv'
        self.extracted = Extract(filename)
        to_float = FloatTransform()
        self.extracted.col[:-1] = to_float(self.extracted.col[:-1])
        # convert class column to integers
        to_int = IntTransform()
        to_int.fit(self.extracted.col[-1])
        self.extracted.col[-1] = to_int(self.extracted.col[-1])

class TestEvaluate(TestAgent):
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
        algorithm = Backpropagation(l_rate, n_epoch)
        agent = Agent(metric=accuracy)
        scores = agent.evaluate(algorithm, loader, n_hidden=n_hidden)
        # scores = evaluate_algorithm(self.dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        # print('Scores: %s' % scores)
        self.assertListEqual(scores, [100.0, 100.0, 100.0, 100.0, 100.0])
        mean_accuracy = sum(scores)/float(len(scores))
        # print('Mean Accuracy: %.3f%%' % (mean_accuracy))
        self.assertEqual(mean_accuracy, 100.000)