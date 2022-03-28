import unittest

from deepton.data.extractor import Dataset
from deepton.data.extractor import Col

class TestCol(unittest.TestCase):
    def test_getitem(self):
        dataset = Dataset('example.csv')
        col = Col(dataset)
        col0 = [
            '2.7810836',
            '1.465489372',
            '3.396561688',
            '1.38807019',
            '3.06407232',
            '7.627531214',
            '5.332441248',
            '6.922596716',
            '8.675418651',
            '7.673756466',
        ]
        self.assertListEqual(col[0], col0)

    def test_setitem(self):
        dataset = Dataset('example.csv')
        col = Col(dataset)
        col_to_set = [
            '2.7810836',
            '1.465489372',
            '3.396561688',
            '1.38807019',
            '3.06407232',
            '7.627531214',
            '5.332441248',
            '6.922596716',
            '8.675418651',
            '7.673756466',
        ]
        col[1] = col_to_set
        self.assertListEqual(col[1], col_to_set)
