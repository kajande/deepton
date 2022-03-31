import unittest
from deepton.data.extractor import Extract

from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.data.analysis import minmax

class TestNormalize(unittest.TestCase):
    def setUp(self) -> None:
        # load and prepare data
        filename = 'example.csv'
        # filename = 'seeds_dataset.csv'
        self.dataset = Extract(filename)
        to_float = FloatTransform()
        for i in range(len(self.dataset[0])-1):
            self.dataset.col[i] = to_float(self.dataset.col[i])
        # convert class column to integers
        to_int = IntTransform().fit(self.dataset.col[-1])
        self.dataset.col[-1] = to_int(self.dataset.col[-1])

class TestFit(TestNormalize):
    def test_simple(self):
        normalize = NormalizeTransform()
        normalize.fit(self.dataset, minmax)
        self.assertListEqual(normalize.borns, minmax(self.dataset))

class TestCall(TestNormalize):
    def test_simple(self):
        normalize = NormalizeTransform()
        normalize.fit(self.dataset, minmax)
        dataset = normalize(self.dataset)
        self.assertGreaterEqual(dataset[0][2], 0)
        self.assertLessEqual(dataset[0][2], 1)
