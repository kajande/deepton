import unittest
from random import seed
from deepton.data.analysis import minmax
from deepton.data.extractor import Extract
from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.model.layer import Layer

from deepton.model.network import Network
from deepton.model.neuron import Neuron
from deepton.model.trainer import Trainer


class TestNetwork(unittest.TestCase):
    def setUp(self) -> None:
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

class TestInit(TestNetwork):
    def test_n_inputs_n_hidden_n_outputs(self):        
        seed(1)
        network = Network(2, 2)
        network.init(n_hidden=1, seed=1)
        expected_network = Network(layers=[
            Layer(neurons=[
                Neuron(weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ])
        self.assertEqual(network.layers, expected_network.layers)

    def test_init(self):
        expected_layers = [
            Layer(neurons=[
                Neuron(weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ]
        expected_network = Network(layers=expected_layers)
        self.assertEqual(expected_network.layers, expected_layers)

class TestEq(TestNetwork):
    def test_true(self):
        net1 = Network(layers = [
            Layer(neurons=[
                Neuron(weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ])
        net2 = Network(layers = [
            Layer(neurons=[
                Neuron(weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ])
        self.assertEqual(net1, net2)

    def test_false(self):
        net1 = Network(layers = [
            Layer(neurons=[
                Neuron(weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ])
        net2 = Network(layers = [ # 1.13436424411240122 vs 0.13436424411240122
            Layer(neurons=[
                Neuron(weights=[1.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ])
        self.assertNotEqual(net1, net2)     

class TestForwardPorpagate(TestNetwork):
    def test_simple(self):
        # test forward propagation
        layers = [
            Layer(neurons=[
                Neuron(weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614])
            ]),
            Layer(neurons=[
                Neuron(weights=[0.2550690257394217, 0.49543508709194095]), 
                Neuron(weights=[0.4494910647887381, 0.651592972722763])
            ])
        ]
        network = Network(layers=layers)
        row = [1, 0, None]
        output = network.forward_propagate(row)
        self.assertListEqual(output, [0.6629970129852887, 0.7253160725279748])

class TestBackwardPropagateError(TestNetwork):
    def test_backward_propagate_error(self):
        # test backpropagation of error
        # print("Testing Backpropagate:")
        layers = [
            Layer(neurons=[
                Neuron(
                    output=0.7105668883115941, 
                    weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614]
                )
            ]),
            Layer(neurons=[
                Neuron(
                    output=0.6213859615555266, 
                    weights=[0.2550690257394217, 0.49543508709194095]
                ), 
                Neuron(
                    output=0.6573693455986976, 
                    weights=[0.4494910647887381, 0.651592972722763]
                )
            ])
        ]
        network = Network(layers=layers)
        expected = [0, 1]
        network.output_errors(expected, [0.6213859615555266, 0.6573693455986976])
        network.output_grads()
        network.backward_propagate_errors()
        network.backward_propagate_grads()

        expected_error_layers = [
            Layer(neurons=[
                Neuron(
                    output=0.7105668883115941, 
                    weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614], 
                    error=0.0026004117552590952,
                    delta=0.0005348048046610517
                )
            ]),
            Layer(neurons=[
                Neuron(
                    output=0.6213859615555266, 
                    weights=[0.2550690257394217, 0.49543508709194095], 
                    error=0.6213859615555266,
                    delta=0.14619064683582808
                ), 
                Neuron(
                    output=0.6573693455986976, 
                    weights=[0.4494910647887381, 0.651592972722763], 
                    error=-0.34263065440130236,
                    delta=-0.0771723774346327
                )
            ])
        ]
        expected_error_network = Network(layers=expected_error_layers)

        self.assertEqual(network, expected_error_network)

class TestUpdateWeights(TestNetwork):

    @unittest.skip("update_weights doesn't have an effect?")
    def test_simple(self):
        layers = [
            Layer(neurons=[
                Neuron(
                    output=0.7105668883115941, 
                    weights=[0.13436424411240122, 0.8474337369372327, 0.763774618976614], 
                    delta=0.0005348048046610517
                )
            ]),
            Layer(neurons=[
                Neuron(
                    output=0.6213859615555266, 
                    weights=[0.2550690257394217, 0.49543508709194095], 
                    delta=0.14619064683582808
                ), 
                Neuron(
                    output=0.6573693455986976, 
                    weights=[0.4494910647887381, 0.651592972722763], 
                    delta=-0.0771723774346327
                )
            ])
        ]
        network = Network(layers=layers)
        row = self.data_extracted[0]
        network.update_weights(row, l_rate=.5)
        print(network.layers)

class TestLearn(TestNetwork):
    def test_simple(self):
        train_data = self.data_extracted
        trainer = Trainer(l_rate=.5, n_epoch=20)
        n_inputs = len(train_data[0]) - 1
        n_outputs = len(set([row[-1] for row in train_data]))
        network = Network(n_inputs=n_inputs, n_outputs=n_outputs)
        network.init(n_hidden=2, seed=1)
        network.learn(train_data, trainer)
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

class TestPredict(TestNetwork):
    def test_predict(self):
        layers = [
            Layer(neurons=[
                Neuron(
                    weights=[-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 
                    output=0.029980305604426185, 
                    delta=0.0059546604162323625
                ), 
                Neuron(
                    weights=[0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 
                    output=0.9456229000211323, 
                    delta=-0.0026279652850863837
                )
            ]),
            Layer(neurons=[
                Neuron(
                    weights=[2.515394649397849, -0.3391927502445985, -0.9671565426390275], 
                    output=0.23648794202357587, 
                    delta=0.04270059278364587
                ), 
                Neuron(
                    weights=[-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 
                    output=0.7790535202438367, 
                    delta=-0.03803132596437354
                )
            ])
        ]        
        network = Network(layers=layers)
        prediction0 = network.predict([2.7810836, 2.550537003, 0])
        self.assertEqual(prediction0, 0)
        prediction1 = network.predict([7.673756466, 3.508563011, 1])
        self.assertEqual(prediction1, 1)

class TestPredictions(TestNetwork):
    def test_simple(self):
        layers = [
            Layer(neurons=[
                Neuron(
                    weights=[-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 
                    output=0.029980305604426185, 
                    delta=0.0059546604162323625
                ), 
                Neuron(
                    weights=[0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 
                    output=0.9456229000211323, 
                    delta=-0.0026279652850863837
                )
            ]),
            Layer(neurons=[
                Neuron(
                    weights=[2.515394649397849, -0.3391927502445985, -0.9671565426390275], 
                    output=0.23648794202357587, 
                    delta=0.04270059278364587
                ), 
                Neuron(
                    weights=[-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 
                    output=0.7790535202438367, 
                    delta=-0.03803132596437354
                )
            ])
        ]  
        network = Network(layers=layers)
        predictions = network.predictions(self.data_extracted)
        self.assertListEqual(predictions, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

if __name__ == '__main__':
    unittest.main()