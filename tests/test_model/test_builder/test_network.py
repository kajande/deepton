import unittest
from random import seed
from deepton.data.extractor import Extract
from deepton.data.transformer import FloatTransform, IntTransform

from deepton.model.builder import Network
from deepton.model.trainer import Trainer


class TestNetwork(unittest.TestCase):
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

class TestInit(TestNetwork):
    def test_n_inputs_n_hidden_n_outputs(self):        
        seed(1)
        network = Network(2, 2)
        network.init(1)
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

class TestEq(TestNetwork):
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

class TestForwardPorpagate(TestNetwork):
    def test_simple(self):
        # test forward propagation
        layers = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
                [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
        network = Network(layers=layers)
        row = [1, 0, None]
        output = network.forward_propagate(row)
        self.assertListEqual(output, [0.6629970129852887, 0.7253160725279748])

class TestBackwardPropagateError(TestNetwork):
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

class TestUpdateWeights(TestNetwork):

    @unittest.skip("update_weights doesn't have an effect?")
    def test_simple(self):
        layers = [
                    [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': 0.0005348048046610517}],
                    [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': -0.0771723774346327}]
                ]
        network = Network(layers=layers)
        row = self.dataset[0]
        network.update_weights(row, l_rate=.5)
        print(network.layers)

class TestLearn(TestNetwork):
    def test_simple(self):
        train_data = self.dataset
        trainer = Trainer(l_rate=.5, n_epoch=20, n_hidden=1)
        # layers = [
        #             [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': 0.0005348048046610517}],
        #             [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': -0.0771723774346327}]
        #         ]
        n_inputs = len(self.dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in self.dataset]))
        network = Network(n_inputs=n_inputs, n_outputs=n_outputs)
        network.learn(train_data, trainer)
        expected_network_layers = [
            [{'weights': [1.2862904647738391, -1.730140232842641, -0.643736514262847], 'output': 0.9414689254772646, 'delta': -0.010492292980594547}], 
            [{'weights': [-1.8273200380301042, 0.6548672618955499], 'output': 0.26568873329611103, 'delta': 0.05183540167555597}, {'weights': [2.2905089827121934, -0.9441520717519927], 'output': 0.7635091702105775, 'delta': -0.04270147412135828}]
        ]
        # print(f"\nTEST LEARN:\n{network.layers}")
        self.assertListEqual(network.layers, expected_network_layers)

class TestPredict(TestNetwork):
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

    def test_predict(self):
        # Test making predictions with the network
        # print("Testing predict:")

        layers = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
            [{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
        network = Network(layers=layers)
        for row in self.dataset:
            prediction = network.predict(row)
            self.assertEqual(prediction, row[-1])
            # print('Expected=%d, Got=%d' % (row[-1], prediction))


if __name__ == '__main__':
    unittest.main()