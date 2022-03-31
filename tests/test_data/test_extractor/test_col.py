import unittest

from deepton.data.extractor import Extract
from deepton.data.extractor import Col

class TestCol(unittest.TestCase):
    pass

class TestInit(TestCol):
    def test_init(self):
        pass

class TestGetitem(TestCol):
    def test_getitem(self):
        dataset = Extract('example.csv')
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

    @unittest.skip("Implement later")
    def test_slice(self):
        dataset = Extract('example.csv')
        col = Col(dataset)
        cols = [
            ['2.7810836',
            '1.465489372',
            '3.396561688',
            '1.38807019',
            '3.06407232',
            '7.627531214',
            '5.332441248',
            '6.922596716',
            '8.675418651',
            '7.673756466',],
            ['2.550537003',
            '2.362125076',
            '4.400293529',
            '1.850220317',
            '3.005305973',
            '2.759262235',
            '2.088626775',
            '1.77106367',
            '-0.242068655',
            '3.508563011']
            ]
        self.assertListEqual(col[0:2], cols)

class TestSetitem(TestCol):
    def test_setitem(self):
        dataset = Extract('example.csv')
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