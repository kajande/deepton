import unittest

from random import seed
from deepton.data.extractor import Extract
from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.model.network import Network
from deepton.model.trainer import Trainer



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
        trainer = Trainer(l_rate, n_epoch, n_hidden)
        scores = trainer.evaluate(self.dataset, n_folds)
        # scores = evaluate_algorithm(self.dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        # print('Scores: %s' % scores)
        self.assertListEqual(scores, [100.0, 100.0, 100.0, 100.0, 100.0])
        mean_accuracy = sum(scores)/float(len(scores))
        # print('Mean Accuracy: %.3f%%' % (mean_accuracy))
        self.assertEqual(mean_accuracy, 100.000)

if __name__ == '__main__':
    unittest.main()