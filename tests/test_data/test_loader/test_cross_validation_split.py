import unittest
from deepton.data.extractor import Extract
from deepton.data.loader import CrossValidationSplitLoader

class TestCrossValidationSplit(unittest.TestCase):
    def test_cross_validation_split(self):
        dataset = Extract('example.csv')
        n_folds = 5
        cross_validation_split = CrossValidationSplitLoader(dataset, n_folds)
        folds = cross_validation_split()

        self.assertEqual(len(folds), n_folds)
        self.assertEqual(len(dataset)//n_folds, len(folds[0]))

    def test_iteration_once(self):
        dataset = Extract('example.csv')
        n_folds = 3
        cross_validation_split = CrossValidationSplitLoader(dataset, n_folds)
        train_set, test_set, validation_set = next(cross_validation_split)

        self.assertEqual(len(train_set)+len(validation_set)+len(dataset)%n_folds, len(dataset))

    def test_iteration_twice(self):
        dataset = Extract('example.csv')
        n_folds = 3
        cross_validation_split = CrossValidationSplitLoader(dataset, n_folds)
        train_set1, test_set1, validation_set1 = next(cross_validation_split)
        train_set2, test_set2, validation_set2 = next(cross_validation_split)
        
        self.assertNotEqual(train_set1, train_set2)


if __name__ == '__main__':
    unittest.main()