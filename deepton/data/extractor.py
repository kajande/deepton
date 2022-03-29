from csv import reader

class Col:
	def __init__(self, dataset):
		self.dataset = dataset

	def __getitem__(self, i):
		return list(map(lambda x: x[i], self.dataset))

	def __setitem__(self, i, value):
		for j in range(len(value)):
			self.dataset[j][i] = value[j]


class Extract: 
	def __init__(self, filename):
		self.data = self._load_csv(filename)
		self.col = Col(self)

	# Load a CSV file
	def _load_csv(self, filename):
		dataset = list()
		with open(filename, 'r') as file:
			csv_reader = reader(file)
			for row in csv_reader:
				if not row:
					continue
				dataset.append(row)
		return dataset

	# def col(self, i):
	# 	return list(map(lambda x: x[i], self.data))

	def __getitem__(self, i):
		return self.data[i]

	def __len__(self):
		return len(self.data)

	def __iter__(self):
		return iter(self.data)