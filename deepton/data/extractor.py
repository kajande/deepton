from csv import reader

class Col:
	def __init__(self, data):
		self.data = data

	def _slice_values(self, slice):
		start, stop, step = slice.start, slice.stop, slice.step
		if not start:
			start = 0
		if not stop:
			stop = len(self.data[0])
		elif stop < 0:
			stop = stop + len(self.data[0])
		if not step:
			step = 1
		return start, stop, step

	def _get_slice(self, slice):
		start, stop, step = self._slice_values(slice)
		cols = []
		for i in range(start, stop, step):
			cols.append(self[i])
		return cols

	def __getitem__(self, j):
		if isinstance(j, slice):
			return self._get_slice(j)
		return list(map(lambda x: x[j], self.data))

	def __setitem__(self, j, value):
		if isinstance(j, slice):
			self._set_slice(j, value)
		else:
			for i in range(len(value)):
				self.data[i][j] = value[i]

	def _set_slice(self, slice, cols):
		start, stop, step = self._slice_values(slice)
		for i in range(start, stop, step):
			self[i] = cols[i]


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
