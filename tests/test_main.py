import unittest

from random import seed
from deepton.main import train_network
from deepton.main import back_propagation, evaluate_algorithm
from deepton.data.extractor import Extract
from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.model.network import Network



# @unittest.skip("Testing IntTransform")
class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        # load and prepare data
        filename = 'example.csv'
        # filename = 'seeds_dataset.csv'
        self.dataset = Extract(filename)
        to_float = FloatTransform()
        for i in range(len(self.dataset[0])-1):
            self.dataset.col[i] = to_float(self.dataset.col[i])
        # convert class column to integers
        to_int = IntTransform().fit(self.dataset.col[-1])
        self.dataset.col[-1] = to_int(self.dataset.col[-1])


    def test_train_network(self):
        # Test training backprop algorithm
        # print("Testing train_network:")
        seed(1)
        n_inputs = len(self.dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in self.dataset]))
        network = Network(n_inputs, 2, n_outputs)
        train_network(network, self.dataset, 0.5, 20, n_outputs)
        expected_layers = [
            [{'weights': [-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 'output': 0.029980305604426185, 'delta': 0.0059546604162323625}, {'weights': [0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 'output': 0.9456229000211323, 'delta': -0.0026279652850863837}],
            [{'weights': [2.515394649397849, -0.3391927502445985, -0.9671565426390275], 'output': 0.23648794202357587, 'delta': 0.04270059278364587}, {'weights': [-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 'output': 0.7790535202438367, 'delta': -0.03803132596437354}]
        ]
        self.assertListEqual(network.layers, expected_layers)

    def test_back_propagation(self):
        # Test Backprop on Seeds dataset
        # print("Testing back_propagation algorithm:")
        seed(1)
        # normalize input variables
        minmax = self.dataset.minmax()
        normalize = NormalizeTransform()
        self.dataset = normalize(self.dataset, minmax)
        # evaluate algorithm
        n_folds = 5
        l_rate = 0.3
        n_epoch = 500
        n_hidden = 5
        scores = evaluate_algorithm(self.dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        # print('Scores: %s' % scores)
        self.assertListEqual(scores, [100.0, 100.0, 100.0, 100.0, 100.0])
        mean_accuracy = sum(scores)/float(len(scores))
        # print('Mean Accuracy: %.3f%%' % (mean_accuracy))
        self.assertEqual(mean_accuracy, 100.000)

if __name__ == '__main__':
    unittest.main()