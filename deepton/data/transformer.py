class FloatTransform:
	# Convert string column to float
	def __call__(self, column):
		column = list(column)
		for i in range(len(column)):
			column[i] = float(column[i].strip())
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
		unique = list(set(class_values))
		unique.sort()
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

	def __call__(self, cols):
		cols = [row[:] for row in cols]
		for row in cols:
			for i in range(len(row)-1):
				row[i] = (row[i] - self._borns[i][0]) / (self._borns[i][1] - self._borns[i][0])
		return cols
		