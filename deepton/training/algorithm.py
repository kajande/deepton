class Backpropagation:
    # Train a network for a fixed number of epochs
    def __init__(self, l_rate, n_epoch):
        self.n_epoch = n_epoch
        self.l_rate = l_rate

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def train(self, network, train_data):
        for epoch in range(self.n_epoch):
            for row in train_data:
                outputs = network.forward_propagate(row)
                expected = [0 for i in range(network.n_outputs)]
                expected[row[-1]] = 1
                network.output_errors(expected, outputs)
                network.output_grads()
                network.backward_propagate_errors()
                network.backward_propagate_grads()
                network.update_weights(row, self.l_rate)