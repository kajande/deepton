import unittest

from deepton.data.extractor import Extract
from deepton.data.transformer import IntTransform

class TestIntTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.data = [['2.7810836','2.550537003','0'],
            ['1.465489372','2.362125076','0'],
            ['3.396561688','4.400293529','0'],
            ['1.38807019','1.850220317','0'],
            ['3.06407232','3.005305973','0'],
            ['7.627531214','2.759262235','1'],
            ['5.332441248','2.088626775','1'],
            ['6.922596716','1.77106367','1'],
            ['8.675418651','-0.242068655','1'],
            ['7.673756466','3.508563011','1']]

class TestInit(TestIntTransform):
    def test_lookup(self):
        to_int = IntTransform()
        self.assertDictEqual(to_int.lookup, {})

class TestCall(TestIntTransform):
    def test_simple(self):
        dataset = Extract('example.csv')
        self.assertIsInstance(dataset.col[-1][0], str)
        to_int = IntTransform().fit(dataset.col[-1])
        dataset.col[-1] = to_int(dataset.col[-1])
        self.assertIsInstance(dataset.col[-1][0], int)
