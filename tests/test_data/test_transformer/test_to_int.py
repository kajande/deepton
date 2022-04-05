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
    def test_lookup_null(self):
        to_int = IntTransform()
        self.assertListEqual(to_int.lookup, [])

    def test_lookup_dict(self):
        to_int = IntTransform({'0': 0, '1': 1})
        self.assertDictEqual(to_int.lookup, {'0': 0, '1': 1})

    def test_lookup_list_null(self):
        to_int = IntTransform([])
        self.assertListEqual(to_int.lookup, [])

    def test_lookup_list_of_one_dict(self):
        to_int = IntTransform([{'0': 0, '1': 1}])
        self.assertDictEqual(to_int.lookup, {'0': 0, '1': 1})

    def test_lookup_exception(self):
        with self.assertRaises(Exception):
            to_int = IntTransform("")
        with self.assertRaises(Exception):
            to_int = IntTransform(0)

# @unittest.skip("")
class TestCall(TestIntTransform):
    def test_call_one_col(self):
        col = ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1']
        expected_col = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        to_int = IntTransform(lookup={'0': 0, '1': 1})
        int_col = to_int(col)
        self.assertListEqual(int_col, expected_col)
        self.assertNotEqual(int_col, col)

    def test_call_one_nested_col(self):
        col = [['0', '0', '0', '0', '0', '1', '1', '1', '1', '1']]
        expected_col = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]

        to_int = IntTransform(lookup={'0': 0, '1': 1})
        int_col = to_int(col)
        self.assertListEqual(int_col, expected_col)
        self.assertNotEqual(int_col, col)

    def test_call_2_cols(self):
        cols = [
            ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1'],
            ['a', 'b', 'c', 'b', 'a', 'c', 'c', 'b', 'b', 'a']
        ]
        expected_cols = [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 2, 1, 0, 2, 2, 1, 1, 0]
        ]

        to_int = IntTransform(lookup=[{'0': 0, '1': 1}, {'a': 0, 'b': 1, 'c': 2}])
        int_cols = to_int(cols)
        self.assertListEqual(int_cols, expected_cols)
        self.assertNotEqual(int_cols, cols)

    # @unittest.skip("")
    def test_call_with_Extract(self):
        extracted = Extract('example.csv')
        to_int = IntTransform(None)
        to_int.fit(extracted.col[-1])
        result_col = to_int(extracted.col[-1])
        self.assertListEqual(result_col, self.expected_col)
        
        self.assertNotEqual(result_col, extracted.col[-1])

        extracted.col[-1] = result_col
        self.assertEqual(extracted.data, self.expected)


class TestFit(TestIntTransform):
    # @unittest.skip("")
    def test_fit(self):
        col = ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1']
        col_copy = list(col)
        to_int = IntTransform([])
        to_int.fit(col)
        self.assertEqual(to_int.lookup, {'0': 0, '1': 1})

        self.assertListEqual(col, col_copy)

    def test_call_2_cols(self):
        cols = [
            ['0', '0', '0', '0', '0', '1', '1', '1', '1', '1'],
            ['a', 'b', 'c', 'b', 'a', 'c', 'c', 'b', 'b', 'a']
        ]
        cols_copy = list(cols)

        to_int = IntTransform()
        to_int.fit(cols)
        self.assertEqual(to_int.lookup, [{'0': 0, '1': 1}, {'a': 0, 'b': 1, 'c': 2}])

        self.assertListEqual(cols, cols_copy)

    def test_fit_with_Extract(self):
        extracted = Extract('example.csv')
        to_int = IntTransform(None)
        to_int.fit(extracted.col[-1])
        self.assertEqual(to_int.lookup, {'0': 0, '1': 1})

if __name__ == '__main__':
    unittest.main()