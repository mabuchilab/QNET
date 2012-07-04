# encoding: utf-8
"""
test_algebra.py

Created by Nikolas Tezak on 2011-02-08.
Copyright (c) 2011 . All rights reserved.
"""

import unittest
from qnet.algebra.operator_algebra import *






class TestOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        id = IdentityOperator
        self.assertEqual(a+0, a)
        self.assertEqual(0+a, a)
        self.assertEqual(1 + a + 1, a + 2)
        self.assertEqual(a + 2, OperatorPlus(ScalarTimesOperator(2,id),a))

    def testAdditionToOperator(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        self.assertEqual(a + b, b + a)
        self.assertEqual(a + b, OperatorPlus(a,b))

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        z = OperatorZero
        self.assertEqual(a-a, z)
        self.assertEqual(a-b, OperatorPlus(a, ScalarTimesOperator(-1,b)))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a + b).space, h1*h2)




class TestOperatorTimes(unittest.TestCase):
    def testIdentity(self):
        h1 = LocalSpace("h1")
        a = OperatorSymbol("a", h1)
        id = IdentityOperator
        self.assertEqual(a * id, a)
        self.assertEqual(id * a, a)

    def testOrdering(self):
         h1 = LocalSpace("h1")
         h2 = LocalSpace("h2")
         a = OperatorSymbol("a", h1)
         b = OperatorSymbol("b", h2)
         c = OperatorSymbol("c", h2)
         self.assertEqual(a * b, OperatorTimes(a,b))

         self.assertEqual(b * a, a * b)
         self.assertEqual(c * a * b * c * a, OperatorTimes(a, a, c, b, c))
    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a * b).space, h1*h2)





class TestScalarTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b =  OperatorSymbol("b", h2)
        z = OperatorZero


        self.assertEqual(a*1,a)
        self.assertEqual(1*a, a)
        self.assertEqual(a*5,ScalarTimesOperator(5, a))
        self.assertEqual(5*a,a*5)
        self.assertEqual(2*a*3, 6*a)
        self.assertEqual(a*5*b, ScalarTimesOperator(5, a*b))
        self.assertEqual(a*(5*b), ScalarTimesOperator(5, a*b))

        self.assertEqual(0 * a, z)
        self.assertEqual(a*0, z)
        self.assertEqual(10 * z, z)

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual((5*(a * b)).space, h1*h2)


class TestLocalOperatorRelations(unittest.TestCase):
    def testNormalOrdering(self):
        h = LocalSpace("h")
        ii = IdentityOperator
        self.assertEqual(Destroy(h) * Create(h) - Create(h) * Destroy(h), ii )

    def testLocalSigmaPi(self):
        h = LocalSpace("h")
        self.assertEqual(LocalSigma(h, 0, 1)* LocalSigma(h, 1, 2), LocalSigma(h, 0, 2))
        self.assertEqual(LocalSigma(h, 0, 0), LocalProjector(h, 0))

    def testAnnihilation(self):
        h = LocalSpace("h")
        z = OperatorZero
        self.assertEqual(Destroy(h) * LocalSigma(h, 0, 1), z)
        self.assertEqual(LocalSigma(h, 1, 0) * Create(h), z)

