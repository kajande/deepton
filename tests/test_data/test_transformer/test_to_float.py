from re import L
import unittest

from deepton.data.extractor import Extract
from deepton.data.transformer import FloatTransform

class TestFloatTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = [
            [2.7810836,2.550537003,'0'],
            [1.465489372,2.362125076,'0'],
            [3.396561688,4.400293529,'0'],
            [1.38807019,1.850220317,'0'],
            [3.06407232,3.005305973,'0'],
            [7.627531214,2.759262235,'1'],
            [5.332441248,2.088626775,'1'],
            [6.922596716,1.77106367,'1'],
            [8.675418651,-0.242068655,'1'],
            [7.673756466,3.508563011,'1']
        ]

class TestCall(TestFloatTransform):
    def test_call(self):
        col = [
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
        to_float = FloatTransform()
        result_col = to_float(col)
        self.assertEqual(result_col, [
            2.7810836,
            1.465489372,
            3.396561688,
            1.38807019,
            3.06407232,
            7.627531214,
            5.332441248,
            6.922596716,
            8.675418651,
            7.673756466,
        ])

        self.assertNotEqual(col, result_col)

    def test_call_2_cols(self):
        cols = [
            ['2.7810836',
            '1.465489372',
            '3.396561688'],
            ['2.550537003',
            '2.362125076',
            '4.400293529']
        ]
        expected_cols = [
            [2.7810836,
            1.465489372,
            3.396561688],
            [2.550537003,
            2.362125076,
            4.400293529]
        ]
        to_float = FloatTransform()
        result_cols = to_float(cols)
        self.assertListEqual(result_cols, expected_cols)
        self.assertNotEqual(cols, result_cols)
        
    def test_with_Extract(self):
        extracted = Extract('example.csv')
        to_float = FloatTransform()
        
        float_col = to_float(extracted.data[0])
        self.assertNotEqual(extracted.data[0], float_col)

        for i in range(len(extracted.data[0])-1):
            extracted.col[i] = to_float(extracted.col[i])
        self.assertListEqual(extracted.data, self.expected)

    def test_with_Extract_2_cols(self):
        extracted = Extract('example.csv')
        to_float = FloatTransform()
        
        float_cols = to_float(extracted.data[:2])
        self.assertNotEqual(extracted.data[:2], float_cols)

        extracted.col[0:2] = to_float(extracted.col[:2])
        self.assertListEqual(extracted.data, self.expected)