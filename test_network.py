from random import seed
from network import initialize_network

import unittest

class TestNetwork(unittest.TestCase):
    def test_initialize_network(self):
        seed(1)
        network = initialize_network(2, 1, 2)
        for layer in network:
            print(layer)


if __name__ == '__main__':
    unittest.main()