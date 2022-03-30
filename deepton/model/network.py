from random import seed
from random import random

class Network:
	# Initialize a network
	def __init__(self, n_inputs, n_hidden, n_outputs):
		self.layers = list()
		hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
		self.layers.append(hidden_layer)
		output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
		self.layers.append(output_layer)