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

        self.expected = [
            [0.1911550432170284, 0.6015484245552352, 0], 
            [0.010623779336795444, 0.5609630674606582, 0], 
            [0.2756134839371173, 1.0, 0], 
            [0.0, 0.45069490252421895, 0], 
            [0.22998792207749216, 0.6995091074953493, 0], 
            [0.856204565678721, 0.6465094215061787, 1], 
            [0.5412628583783597, 0.5020494605166291, 1], 
            [0.759470547568756, 0.4336439608133772, 1], 
            [1.0, 0.0, 1], 
            [0.8625477853350042, 0.8079144877852557, 1]
        ]

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
        self.assertListEqual(dataset.data, self.expected)