import unittest
from random import seed

from deepton.model.network import Network

class TestInit(unittest.TestCase):
    def test_n_inputs_n_hidden_n_outputs(self):        
        seed(1)
        network = Network(2, 1, 2)
        expected_layers = [
            [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
        ]
        self.assertEqual(network.layers, expected_layers)

    def test_init(self):
        expected_layers = [
            [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
        ]
        expected_network = Network(layers=expected_layers)
        self.assertEqual(expected_network.layers, expected_layers)

class TestEq(unittest.TestCase):
    def test_true(self):
        net1 = Network(layers = [
            [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
        ])
        net2 = Network(layers = [
            [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
        ])
        self.assertEqual(net1, net2)

    def test_false(self):
        net1 = Network(layers = [
            [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
        ])
        net2 = Network(layers = [ # 1.13436424411240122 vs 0.13436424411240122
            [{'weights': [1.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
            [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
        ])
        self.assertNotEqual(net1, net2)     

if __name__ == '__main__':
    unittest.main()