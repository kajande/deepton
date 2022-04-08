from csv import reader

class Col:
	def __init__(self, data):
		self.data = data

	def __getitem__(self, j):
		return [row[j] for row in self.data]
		# return list(map(lambda x: x[j], self.data))

	def __setitem__(self, j, value):
		for i in range(len(value)):
			self.data[i][j] = value[i]


class Extract: 
	def __init__(self, filename):
		self.data = self._load_csv(filename)
		self._col = Col(self.data)

	@property
	def col(self):
		return self._col

	# Load a CSV file
	def _load_csv(self, filename):
		data = list()
		with open(filename, 'r') as file:
			csv_reader = reader(file)
			for row in csv_reader:
				if not row:
					continue
				data.append(row)
		return data


class ExtractList(Extract):
	def __init__(self, list_data):
		self.data = list_data
		self._col = Col(self.data)
