import unittest
from deepton.data.extractor import Extract

class TestExtract(unittest.TestCase):
    def setUp(self) -> None:        
        self.expected = [['2.7810836','2.550537003','0'],
            ['1.465489372','2.362125076','0'],
            ['3.396561688','4.400293529','0'],
            ['1.38807019','1.850220317','0'],
            ['3.06407232','3.005305973','0'],
            ['7.627531214','2.759262235','1'],
            ['5.332441248','2.088626775','1'],
            ['6.922596716','1.77106367','1'],
            ['8.675418651','-0.242068655','1'],
            ['7.673756466','3.508563011','1']]


class TestInit(TestExtract):
    def test_init(self):
        extracted = Extract('example.csv')
        self.assertListEqual(extracted.data, self.expected)

class TestGetim(TestExtract):
    def test_getitem(self):
        extracted = Extract('example.csv')
        elm = extracted[0]
        self.assertListEqual(elm, self.expected[0])

class TestIter(TestExtract):
    def test_iter(self):
        extracted = Extract('example.csv')
        for elm in extracted:
            pass
        self.assertListEqual(elm, self.expected[-1])

class TestLen(TestExtract):
    def test_len(self):
        extracted = Extract('example.csv')
        self.assertEqual(len(extracted), len(self.expected))

class TestCol(TestExtract):
    def test_col(self):
        extracted = Extract('example.csv')
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
        self.assertListEqual(extracted.col[0], col0)