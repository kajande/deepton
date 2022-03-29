import unittest
from deepton.data.extractor import Extract

class TestExtract(unittest.TestCase):
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

    def test_init(self):
        dataset = Extract('example.csv')
        self.assertListEqual(dataset.data, self.data)

    def test_getitem(self):
        dataset = Extract('example.csv')
        elm = dataset[0]
        self.assertListEqual(elm, self.data[0])

    def test_iter(self):
        dataset = Extract('example.csv')
        for elm in dataset:
            pass
        self.assertListEqual(elm, self.data[-1])

    def test_len(self):
        dataset = Extract('example.csv')
        self.assertEqual(len(dataset), len(self.data))

    def test_col(self):
        dataset = Extract('example.csv')
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
        self.assertListEqual(dataset.col[0], col0)