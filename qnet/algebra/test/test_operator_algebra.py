# encoding: utf-8
"""
test_algebra.py

Created by Nikolas Tezak on 2011-02-08.
Copyright (c) 2011 . All rights reserved.
"""

import unittest
from qnet.algebra.operator_algebra import *


class TestOperatorCreation(unittest.TestCase):
    def testImplicitHilbertSpace(self):
        hs = local_space("hs")
        h2 = local_space("h2")
        aa = OperatorSymbol("aa",hs)
        bb = OperatorSymbol("aa",hs*h2)
        a = Destroy(hs)
        self.assertEqual(aa, OperatorSymbol("aa","hs"))
        self.assertEqual(bb, OperatorSymbol("aa",("hs","h2")))
        self.assertEqual(a, Destroy("hs"))

    def testIdentity(self):
        self.assertEqual(Create("1"), Create("1"))

    def testMatch(self):
        A = wc("A", head = Operator)
        a = Create("1")
        c = Create("1")
        b = OperatorSymbol("b","hs")
        b2 = OperatorSymbol("b","hs")
        self.assertEqual(a,c)
        self.assertEqual(b,b2)
        self.assertEqual(match(A, a),Match(A = a))
        self.assertEqual(match(A, b),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((b, b))),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((b, b2))),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((a, c))),Match(A = a))



class TestOperatorAddition(unittest.TestCase):



    def testAdditionToScalar(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        id = IdentityOperator
        self.assertEqual(a+0, a)
        self.assertEqual(0+a, a)
        self.assertEqual(1 + a + 1, a + 2)
        self.assertEqual(a + 2, OperatorPlus(ScalarTimesOperator(2,id),a))


    def testAdditionToOperator(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        self.assertEqual(a + b, b + a)
        self.assertEqual(a + b, OperatorPlus(a,b))

    def testSubtraction(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        z = OperatorZero
        self.assertEqual(a-a, z)
        self.assertEqual(a-b, OperatorPlus(a, ScalarTimesOperator(-1,b)))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a + b).space, h1*h2)




class TestOperatorTimes(unittest.TestCase):
    def testIdentity(self):
        h1 = local_space("h1")
        a = OperatorSymbol("a", h1)
        id = IdentityOperator
        self.assertEqual(a * id, a)
        self.assertEqual(id * a, a)

    def testOrdering(self):
         h1 = local_space("h1")
         h2 = local_space("h2")
         a = OperatorSymbol("a", h1)
         b = OperatorSymbol("b", h2)
         c = OperatorSymbol("c", h2)
         self.assertEqual(a * b, OperatorTimes(a,b))

         self.assertEqual(b * a, a * b)
         self.assertEqual(c * a * b * c * a, OperatorTimes(a, a, c, b, c))
    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a * b).space, h1*h2)





class TestScalarTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        b =  OperatorSymbol("b", h2)
        z = OperatorZero

        self.assertEqual(a+a,2*a)
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

    def testScalarCombination(self):
        a = OperatorSymbol("a", "h1")
        self.assertEqual(a+a, 2*a)
        self.assertEqual(3*a + 4*a, 7 * a)
        self.assertEqual(Create("1") + Create("1"), 2 * Create("1"))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual((5*(a * b)).space, h1*h2)


class TestLocalOperatorRelations(unittest.TestCase):
    def testNormalOrdering(self):
        h = local_space("h")
        ii = IdentityOperator
        self.assertEqual(Destroy(h) * Create(h) - Create(h) * Destroy(h), ii )

    def testLocalSigmaPi(self):
        h = local_space("h")
        self.assertEqual(LocalSigma(h, 0, 1)* LocalSigma(h, 1, 2), LocalSigma(h, 0, 2))
        self.assertEqual(LocalSigma(h, 0, 0), LocalProjector(h, 0))

    def testAnnihilation(self):
        h = local_space("h")
        z = OperatorZero
        self.assertEqual(Destroy(h) * LocalSigma(h, 0, 1), z)
        self.assertEqual(LocalSigma(h, 1, 0) * Create(h), z)


class TestOperatorMatrices(unittest.TestCase):
    def testConstruction(self):
        h1, h2, h3 = local_space("h1"), local_space("h2"), local_space("h3")
        a, b, c = Destroy(h1), Destroy(h2), Destroy(h3)
        self.assertEqual(np_conjugate(a), a.dag())

        M = OperatorMatrix([[a,b],[c,a]])
#        self.assertEqual(M.matrix, np_array([[a,b],[b,a]]))
        self.assertEqual(M, OperatorMatrix(np_array([[a,b],[c,a]])))
        self.assertEqual(M.T, OperatorMatrix(np_array([[a,c],[b,a]])))
        self.assertEqual(M.conjugate(), OperatorMatrix(np_array([[a.dag(),b.dag()],[c.dag(),a.dag()]])))
        self.assertEqual(M.H, OperatorMatrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]])))
        self.assertEqual(M.H, OperatorMatrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]])))

    def testMathOperations(self):
        M = OperatorMatrix([[Create("1"), 0],[0, Destroy("1")]])
        N = OperatorMatrix([[Destroy("1"), Create("2")],[0, Destroy("1")]])
        self.assertEqual(M+N, OperatorMatrix([[Create("1")+Destroy("1"), Create("2")],[0, 2*Destroy("1")]]))
        self.assertEqual(M*N, OperatorMatrix([[Create("1")*Destroy("1"), Create("1")*Create("2")],[0, Destroy("1")*Destroy("1")]]))
        self.assertEqual(IdentityOperator * M, M)
        self.assertEqual(1 * M, M)
        self.assertEqual(Create("1") * identity_matrix(2), OperatorMatrix([[Create("1"),0],[0,Create("1")]]))


