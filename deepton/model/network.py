import random

from deepton.model.layer import Layer


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
        hidden_layer = Layer(self.n_inputs, n_hidden)
        output_layer = Layer(n_hidden, self.n_outputs)
        self._layers.extend([hidden_layer, output_layer])

    @property
    def layers(self):
        return self._layers

    def __eq__(self, other):
        return self.layers == other.layers

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        for layer in self._layers:
            row = layer.forward_propagate(row[:])
        return row

    def output_errors(self, expected, outputs):
        self.layers[-1].output_errors(expected, outputs)

    def output_grads(self):
        self.layers[-1].backward_propagate_grads()

    # Backpropagate error and store in neurons
    def backward_propagate_errors(self):
        for i in reversed(range(len(self._layers)-1)):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]
            layer.backward_propagate_errors(next_layer)

    def backward_propagate_grads(self):
        for layer in self.layers[:-1]:
            layer.backward_propagate_grads()

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i, layer in enumerate(self._layers):
            if i == 0:
                inputs = row[:-1]
            else:
                inputs = [neuron.output for neuron in self._layers[i - 1]]
            layer.update_weights(inputs, l_rate)

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
                self.output_errors(expected, outputs)
                self.output_grads()
                self.backward_propagate_errors()
                self.backward_propagate_grads()
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