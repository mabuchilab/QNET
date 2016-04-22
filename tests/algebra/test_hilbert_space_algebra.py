#This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QNET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QNET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

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
        self.assertRaises(BasisNotSetError,lambda : h3.dimension)
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

    def testOperations(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        h3 = local_space("h3")

        h123 = h1 * h2 * h3
        h12 = h1 * h2
        h23 = h2 * h3
        h13 = h1 * h3
        self.assertEqual(h12 * h13, h123)
        self.assertEqual(h12 / h13, h2)
        self.assertEqual(h12 & h13, h1)
        self.assertEqual((h12 / h13) * (h13 & h12), h12)
        self.assertEqual(h1 & h12, h1)

