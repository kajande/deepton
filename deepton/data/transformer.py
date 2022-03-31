class FloatTransform:
	# Convert string column to float
	def __call__(self, column):
		# for row in column:
		for i in range(len(column)):
			column[i] = float(column[i].strip())
			# row[self.column] = float(row[self.column].strip())
		return column

class IntTransform:
	# Convert string column to integer
	def __init__(self, lookup=None):
		if lookup:
			self.lookup = lookup
		else:
			self.lookup = dict()

	def fit(self, column):
		class_values = column
		unique = set(class_values)
		for i, value in enumerate(unique):
			self.lookup[value] = i
		return self

	def __call__(self, column):
		return [self.lookup[row] for row in column]


# Rescale dataset columns to the range 0-1
class NormalizeTransform:
	def __init__(self, borns=None):
		self._borns = borns

	@property
	def borns(self):
		return self._borns

	def fit(self, dataset, method):
		self._borns = method(dataset)

	def __call__(self, dataset):
		for row in list(dataset):
			for i in range(len(row)-1):
				row[i] = (row[i] - self._borns[i][0]) / (self._borns[i][1] - self._borns[i][0])
		return dataset