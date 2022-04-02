from csv import reader

class Col:
	def __init__(self, dataset):
		self.dataset = dataset

	def _get_slice(self, slice):
		start, stop, step = slice.start, slice.stop, slice.step
		if not slice.step:
			step = 1
		cols = []
		for i in range(start, stop, step):
			cols.append(self[i])
		return cols

	def __getitem__(self, j):
		if isinstance(j, slice):
			return self._get_slice(j)
		return list(map(lambda x: x[j], self.dataset))

	def __setitem__(self, j, value):
		for i in range(len(value)):
			self.dataset[i][j] = value[i]


class Extract: 
	def __init__(self, filename):
		self.data = self._load_csv(filename)
		self._col = Col(self.data)

	@property
	def col(self):
		return self._col

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