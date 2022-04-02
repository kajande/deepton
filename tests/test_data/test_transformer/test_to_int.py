from copy import copy
import unittest

from deepton.data.extractor import Extract
from deepton.data.transformer import IntTransform

class TestIntTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = [
            ['2.7810836','2.550537003',0],
            ['1.465489372','2.362125076',0],
            ['3.396561688','4.400293529',0],
            ['1.38807019','1.850220317',0],
            ['3.06407232','3.005305973',0],
            ['7.627531214','2.759262235',1],
            ['5.332441248','2.088626775',1],
            ['6.922596716','1.77106367',1],
            ['8.675418651','-0.242068655',1],
            ['7.673756466','3.508563011',1]
        ]
        self.expected_col = [row[-1] for row in self.expected]

class TestInit(TestIntTransform):
    def test_none(self):
        to_int = IntTransform()
        self.assertDictEqual(to_int.lookup, {})
    
    def test_not_none(self):
        to_int = IntTransform({'0': 0, '1': 1})
        self.assertDictEqual(to_int.lookup, {'0': 0, '1': 1})
        self.assertEqual(to_int.lookup['0'], 0)
        self.assertEqual(to_int.lookup['1'], 1)

class TestFit(TestIntTransform):
    def test_fit(self):
        col = ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1']
        col_copy = list(col)
        to_int = IntTransform()
        to_int.fit(col)
        self.assertTrue(to_int.lookup=={'0': 0, '1': 1})
        self.assertListEqual(col, col_copy)


class TestCall(TestIntTransform):
    def test_call(self):
        col = ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1']
        expected_col = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        to_int = IntTransform(lookup={'0': 0, '1': 1})
        int_col = to_int(col)
        self.assertListEqual(int_col, expected_col)
        self.assertNotEqual(int_col, col)

    def test_call_with_Extract(self):
        extracted = Extract('example.csv')
        to_int = IntTransform().fit(extracted.col[-1])
        result_col = to_int(extracted.col[-1])
        self.assertListEqual(result_col, self.expected_col)
        
        self.assertNotEqual(result_col, extracted.col[-1])

        extracted.col[-1] = result_col
        self.assertEqual(extracted.data, self.expected)
