import random
from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

class Network:
    # Initialize a network
    def __init__(self, n_inputs=None, n_outputs=None, layers=None):
        # if layers:
        #     self.layers = layers
        # else:
        #     self.layers = self.from_layers(n_inputs, n_hidden, n_outputs)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        if layers is None:
            self._layers = []
        else:
            self._layers = layers

        # if not layers:
        #     layers = self.from_layers(n_inputs, n_hidden, n_outputs)
        # self._layers = layers

    def init(self, n_hidden, seed):
        # Don't call this method if self._layers are already initialized (exist)
        # if self._layers:
        #     raise Exception("This model already has layers")
        random.seed(seed)
        hidden_layer = self.layer_init(self.n_inputs, n_hidden)
        output_layer = self.layer_init(n_hidden, self.n_outputs)
        self._layers.extend([hidden_layer, output_layer])

    def layer_init(self, n_inputs, n_outputs):
        return [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_outputs)]

    @property
    def layers(self):
        return self._layers

    def __eq__(self, other):
        return self.layers == other.layers

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def layer_forward_propagate(self, layer, inputs):
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        return new_inputs

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self._layers:
            inputs = self.layer_forward_propagate(layer, inputs)
        return inputs


    def output_layer_backward_propagate_error(self, layer, expected):
        errors = list()
        for j in range(len(layer)):
            neuron = layer[j]
            errors.append(neuron['output'] - expected[j])
        for neuron, error in zip(layer, errors):
            neuron['delta'] = error * transfer_derivative(neuron['output'])

    def layer_backward_propagate_error(self, layer, next_layer):
            errors = list()
            for j in range(len(layer)):
                error = 0.0
                for neuron in next_layer:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
            for neuron, error in zip(layer, errors):
                neuron['delta'] = error * transfer_derivative(neuron['output'])

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self))):
            layer = self._layers[i]
            if i == len(self)-1: # last layer: output layer
                self.output_layer_backward_propagate_error(layer, expected)
            else:
                next_layer = self._layers[i + 1]
                self.layer_backward_propagate_error(layer, next_layer)

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self[i - 1]]
            for neuron in self[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= l_rate * neuron['delta']

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def learn(self, train_data, trainer):
        # trainer.train(self)
        # self.init(trainer.n_hidden, trainer.seed) # later refactor to `trainer.initializer` instead
        for epoch in range(trainer.n_epoch):
            for row in train_data:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                # print(f"\n\nn_outputs: {trainer.n_outputs}\n\n")
                expected[row[-1]] = 1
                self.backward_propagate_error(expected)
                self.update_weights(row, trainer.l_rate)
        # update here the `trainer.initializer` parameters

    # Make a prediction with a network
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


    def predictions(self, test):
        predictions = list()
        for row in test:
            prediction = self.predict(row)
            predictions.append(prediction)
        return(predictions)