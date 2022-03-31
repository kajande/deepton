import unittest
from deepton.data.analysis import minmax

class TestMinMax(unittest.TestCase):
    def setUp(self):
        self.dataset = [[.5, -.2, 1.3], [2.1, -1.0, 1.07], [1.5, -2.1, 0.5], [-1.1, 2.1, .2]]

class TestCall(TestMinMax):
   def test_call(self):
       borns = minmax(self.dataset)
       self.assertListEqual(borns, [[-1.1, 2.1], [-2.1, 2.1], [.2, 1.3]])
