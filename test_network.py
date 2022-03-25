from random import seed
from network import initialize_network, forward_propagate, backward_propagate_error

import unittest

class TestNetwork(unittest.TestCase):
    def test_initialize_network(self):
        seed(1)
        network = initialize_network(2, 1, 2)
        for layer in network:
            print('layer:', layer)

    def test_forward_propagate(self):
        # test forward propagation
        network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
                [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
        row = [1, 0, None]
        output = forward_propagate(network, row)
        print()
        print('output:', output)

    def test_backward_propagate_error(self):
        # test backpropagation of error
        print("Testing Backpropagate:")
        network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
                [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
        expected = [0, 1]
        backward_propagate_error(network, expected)
        for layer in network:
            print(layer)

if __name__ == '__main__':
    unittest.main()