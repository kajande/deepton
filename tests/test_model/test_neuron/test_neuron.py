import random
import unittest

from deepton.model.neuron import Neuron


class TestNeuron(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1)
        self.data_extracted = [
            [2.7810836,2.550537003,0],
            [1.465489372,2.362125076,0],
            [3.396561688,4.400293529,0],
            [1.38807019,1.850220317,0],
            [3.06407232,3.005305973,0],
            [7.627531214,2.759262235,1],
            [5.332441248,2.088626775,1],
            [6.922596716,1.77106367,1],
            [8.675418651,-0.242068655,1],
            [7.673756466,3.508563011,1]
        ]

class TestInit(TestNeuron):
    def test_inputs_None(self):
        neuron = Neuron()
        self.assertIsNone(neuron.weights)

    def test_inputs_not_None(self):
        neuron = Neuron(n_inputs=2)
        expected = [0.13436424411240122, 0.8474337369372327, 0.763774618976614]
        self.assertListEqual(neuron.weights, expected)

class TestEq(TestNeuron):
    def test_eq(self):
        random.seed(1)
        neuron1 = Neuron(n_inputs=2)
        random.seed(1)
        neuron2 = Neuron(n_inputs=2)
        self.assertEqual(neuron1, neuron2)

class TestForwardPropagate(TestNeuron):
    def test_basic(self):
        inputs = [1, 0, None]
        neuron = Neuron(n_inputs=2)
        neuron.forward_propagate(inputs)
        self.assertEqual(neuron.activation, 0.8981388630890152)

class TestActivate(TestNeuron):
    def test_basic(self):
        neuron = Neuron(n_inputs=2)
        neuron.activation = 0.8981388630890152
        neuron.activate()
        self.assertEqual(neuron.output, 0.7105668883115941)