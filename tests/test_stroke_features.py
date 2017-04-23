import unittest
import doctest

import numpy as np

import enviroment
from smartmanuscript import stroke_features


class TestStrokeFeatures(unittest.TestCase):
    def test_fix_corruption(self):
        strokes = [np.array([[0, 0], [0, 0], [1, 0]]),
                   np.array([[1, 0], [2, 0]]),
                   np.array([[2, 1], [3, 1]])]
        ink = stroke_features.Ink.from_corrupted_stroke(strokes)
        self.assertEqual(len(ink.strokes), 2)
        self.assertEqual(len(ink.strokes[0]), 3)
        self.assertEqual(len(ink.strokes[1]), 2)


if __name__ == "__main__":
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.makeSuite(TestStrokeFeatures))
    test_suite.addTest(doctest.DocTestSuite(stroke_features))
    unittest.TextTestRunner().run(test_suite)
