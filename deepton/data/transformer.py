
class FloatTransform:
	# Convert string column to float
	def _call_one_column(self, column):
		column = list(column)
		for i in range(len(column)):
			column[i] = float(column[i].strip())
		return column

	def __call__(self, cols):
		if isinstance(cols[0], list):
			nested_cols = list(cols)
		else:
			nested_cols = [cols]
		results = [self._call_one_column(col) for col in nested_cols]
		if isinstance(cols[0], list):
			return results
		else:
			return results[0]

class IntTransform:
	# Convert string column to integer
	def __init__(self, lookup=None):
		# print("INSIDE INIT:\n")
		# print(f"\narg_lookup:{lookup}\n")
		if lookup is None:
			lookup = []
		self.lookup = lookup
		# print(f"\nself._lookup:{self._lookup}\n")
		# print("OUTSIDE INIT:\n")


	@property
	def lookup(self):
		if len(self._lookup) == 1:
			return self._lookup[0]
		return self._lookup

	@lookup.setter
	def lookup(self, val):
		if isinstance(val, dict):
			self._lookup = [val]
		elif isinstance(val, list):
			self._lookup = val
		else:
			raise Exception("Argument must be a `dict` or a `list` ")

	def fit(self, cols):
		if not isinstance(cols[0], list):
			cols = [cols]
		for col in cols:
			lookup = {}
			unique = list(set(col))
			unique.sort()
			for i, value in enumerate(unique):
				lookup[value] = i
			self._lookup.append(lookup)

	def __call__(self, cols):
		if isinstance(cols[0], list): # not nested
			nested_cols = list(cols)
		else:
			nested_cols = [cols]
		result = [[lookup[row] for row in col] for lookup, col in zip(self._lookup, nested_cols)]
		if isinstance(cols[0], list): # not nested
			return result
		else:
			return result[0]


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
		