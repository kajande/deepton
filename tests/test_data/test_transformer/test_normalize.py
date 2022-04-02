import unittest
from deepton.data.extractor import Extract

from deepton.data.transformer import FloatTransform, IntTransform, NormalizeTransform
from deepton.data.analysis import minmax

# @unittest.skip("Testing IntTransform..")
class TestNormalize(unittest.TestCase):
    def setUp(self) -> None:
        # load and prepare data
        filename = 'example.csv'
        # filename = 'seeds_dataset.csv'
        self.extracted = Extract(filename)
        to_float = FloatTransform()
        for i in range(len(self.extracted.data[0])-1):
            self.extracted.col[i] = to_float(self.extracted.col[i])
        # convert class column to integers
        to_int = IntTransform()
        to_int.fit(self.extracted.col[-1])
        self.extracted.col[-1] = to_int(self.extracted.col[-1])

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
    def test_fit(self):
        normalize = NormalizeTransform()
        normalize.fit(self.extracted.data, minmax)
        self.assertListEqual(normalize.borns, minmax(self.extracted.data))

        self.assertNotEqual(self.extracted.data, self.expected)


class TestCall(TestNormalize):
    def test_call(self):
        cols = [
            [3, 0],
            [5, 0], 
            [1, 0], 
            [0, 0], 
            [9, 0], 
            [8, 1],
            [10, 1],
            [7, 1],
            [2, 1],
            [6, 1]
        ]
        normalize = NormalizeTransform(borns=[[1, 3]])
        normalized_cols = normalize(cols)
        self.assertEqual(normalized_cols, [
            [1, 0], 
            [2, 0], 
            [0, 0], 
            [-.5, 0], 
            [4, 0], 
            [3.5, 1], 
            [4.5, 1], 
            [3, 1], 
            [.5, 1], 
            [2.5, 1]
        ])

    # @unittest.skip("Test simple test_call first")
    def test_call_with_fit(self):
        normalize = NormalizeTransform()
        normalize.fit(self.extracted.data, minmax)
        self.assertNotEqual(self.extracted.data, self.expected)

        normalized = normalize(self.extracted.data)
        self.assertListEqual(normalized, self.expected)

        self.assertNotEqual(self.extracted.data, self.expected)