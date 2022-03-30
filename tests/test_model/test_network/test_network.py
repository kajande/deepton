import unittest
from random import seed
from deepton.model import network

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

class TestForwardPorpagate(unittest.TestCase):
    def test_simple(self):
        # test forward propagation
        layers = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
                [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
        network = Network(layers=layers)
        row = [1, 0, None]
        output = network.forward_propagate(row)
        self.assertListEqual(output, [0.6629970129852887, 0.7253160725279748])

class TestBackwardPropagateError(unittest.TestCase):
    def test_backward_propagate_error(self):
        # test backpropagation of error
        # print("Testing Backpropagate:")
        layers = [
                    [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
                    [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]
                ]
        network = Network(layers=layers)
        expected = [0, 1]
        network.backward_propagate_error(expected)

        expected_error_layers = [
                                    [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': 0.0005348048046610517}],
                                    [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': -0.0771723774346327}]
                                ]
        expected_error_network = Network(layers=expected_error_layers)
        # print("\nERROR NETWORK:")
        # for layer in layers:
        #     print(layer)
        self.assertEqual(network, expected_error_network)


if __name__ == '__main__':
    unittest.main()