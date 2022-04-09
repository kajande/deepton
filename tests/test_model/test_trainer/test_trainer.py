import unittest
import random
from deepton.data.analysis import minmax
from deepton.data.loader import CrossValidationSplitLoader
from deepton.model.layer import Layer
from deepton.model.neuron import Neuron
from deepton.training.metrics import accuracy

from deepton.model.network import Network
from deepton.data.extractor import Extract
from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.model.trainer import Trainer

class TestTrainer(unittest.TestCase):
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

class TestTrain(TestTrainer):
    def test_simple(self):
        train_data = self.extracted.data
        trainer = Trainer(l_rate=.5, n_epoch=20)
        n_inputs = len(train_data[0]) - 1
        n_outputs = len(set([row[-1] for row in train_data]))
        network = Network(layers=[
            Layer(n_inputs=n_inputs, n_outputs=2),
            Layer(n_inputs=2, n_outputs=n_outputs)
        ])
        # network.init(n_hidden=2)
        # network.learn(train_data, trainer)
        trainer.train(network, train_data)
        expected_layers = [
            Layer(neurons=[
                Neuron(
                    weights=[-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 
                    output=0.029980305604426185, 
                    error=0.2047577704930848,
                    delta=0.0059546604162323625
                ), 
                Neuron(
                    weights=[0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 
                    output=0.9456229000211323, 
                    error=-0.05110761338809097,
                    delta=-0.0026279652850863837
                )
            ]),
            Layer(neurons=[
                Neuron(
                    weights=[2.515394649397849, -0.3391927502445985, -0.9671565426390275], 
                    output=0.23648794202357587, 
                    error=0.23648794202357587,
                    delta=0.04270059278364587
                ), 
                Neuron(
                    weights=[-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 
                    output=0.7790535202438367, 
                    error=-0.22094647975616333,
                    delta=-0.03803132596437354
                )
            ])
        ]
        expected_network = Network(layers=expected_layers)
        self.assertListEqual(network.layers, expected_network.layers)


class TestEvaluate(TestTrainer):
    def test_evaluate(self):
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
        trainer = Trainer(l_rate, n_epoch)
        scores = trainer.evaluate(loader, accuracy, n_hidden)
        # scores = evaluate_algorithm(self.dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        # print('Scores: %s' % scores)
        self.assertListEqual(scores, [100.0, 100.0, 100.0, 100.0, 100.0])
        mean_accuracy = sum(scores)/float(len(scores))
        # print('Mean Accuracy: %.3f%%' % (mean_accuracy))
        self.assertEqual(mean_accuracy, 100.000)