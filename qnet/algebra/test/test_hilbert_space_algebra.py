
from qnet.algebra.hilbert_space_algebra import *
import unittest

class TestHilbertSpaces(unittest.TestCase):

    def testProductSpace(self):

        # create HilbertSpaces
        h1 = local_space("h1")
        h2 = local_space("h2")
        h3 = local_space("h3")

        # productspace
        self.assertEqual(h1 * h2, ProductSpace(h1, h2))
        self.assertEqual(h3 * h1 * h2, ProductSpace(h1, h2, h3))

        # space "subtraction/division/cancellation"
        self.assertEqual((h1 * h2) / h1, h2)
        self.assertEqual((h1 * h2 * h3) / h1, h2 * h3)
        self.assertEqual((h1 * h2 * h3) / (h1 * h3), h2)

        # space "intersection"
        self.assertEqual((h1 * h2) & h1, h1)
        self.assertEqual((h1 * h2 * h3) & h1, h1)
        self.assertEqual(h1 * h1, h1)


    def testDimension(self):
        h1 = local_space("h1", dimension = 10)
        h2 = local_space("h2", dimension = 20)
        h3 = local_space("h3")
        h4 = local_space("h4", dimension = 100)

        self.assertEqual((h1*h2).dimension, h1.dimension * h2.dimension)
        self.assertEqual(h3.dimension, inf)
        self.assertEqual(h4.dimension, 100)


    def testSpaceOrdering(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        h3 = local_space("h3")

        self.assertTrue(h1 <= h1)
        self.assertTrue(h1 <= (h1 * h2))
        self.assertFalse(h1 <= h2)
        self.assertFalse(h1 < h1)
        self.assertTrue(TrivialSpace < h1 < FullSpace)
        self.assertTrue(h1>= h1)
        self.assertTrue(h1 * h2 > h2)
        self.assertFalse(h1 * h2 > h3)
