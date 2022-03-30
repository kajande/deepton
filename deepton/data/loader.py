from random import randrange

# Split a dataset into k folds
class CrossValidationSplitLoader:
	def __init__(self, dataset, n_folds):
		self._folds = self._compute(dataset, n_folds)
		self._i = -1
		self._n_folds = n_folds

	def _compute(self, dataset, n_folds):
		dataset_split = list()
		dataset_copy = list(dataset)
		fold_size = int(len(dataset) / n_folds)
		for i in range(n_folds):
			fold = list()
			while len(fold) < fold_size:
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
			dataset_split.append(fold)
		return dataset_split

	def __call__(self):
		return self._folds

	def __iter__(self):
		return self

	def __next__(self):
		# for validation_set in self._folds:
		self._i += 1
		if self._i >= self._n_folds:
			raise StopIteration
		train_set = list(self._folds)
		validation_set = train_set.pop(self._i)
		train_set = sum(train_set, [])
		test_set = list()
		for row in validation_set:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		return train_set, test_set, validation_set