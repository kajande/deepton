class ToFloat:
	# Convert string column to float
	def __call__(self, column):
		# for row in column:
		for i in range(len(column)):
			column[i] = float(column[i].strip())
			# row[self.column] = float(row[self.column].strip())
		return column

class ToInt:
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
