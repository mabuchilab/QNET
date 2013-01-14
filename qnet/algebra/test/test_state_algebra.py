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


import unittest
from qnet.algebra.state_algebra import *



class TestStateAddition(unittest.TestCase):

    def testAdditionToZero(self):
        hs = local_space("hs")
        a = KetSymbol("a", hs)
        z = KetZero
        self.assertEqual(a+z, a)
        self.assertEqual(z+a, a)
        self.assertEqual(z+z, z)
        self.assertEqual(z, 0)


    def testAdditionToOperator(self):
        hs = local_space("hs")
        a = KetSymbol("a", hs)
        b = KetSymbol("b", hs)
        self.assertEqual(a + b, b + a)
        self.assertEqual(a + b, KetPlus(a,b))

    def testSubtraction(self):
        hs = local_space("hs")
        a = KetSymbol("a", hs)
        b = KetSymbol("b", hs)
        z = KetZero
        self.assertEqual(a-a, z)
        self.assertEqual(a-b, KetPlus(a, ScalarTimesKet(-1,b)))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        self.assertRaises(UnequalSpaces, a.__add__, b)


    def testEquality(self):
        h1 = local_space("h1")
        self.assertEqual(CoherentStateKet(h1, 10.)+CoherentStateKet(h1, 20.), CoherentStateKet(h1, 20.)+CoherentStateKet(h1, 10.))





class TestTensorKet(unittest.TestCase):

    def testIdentity(self):
        h1 = local_space("h1")
        a = KetSymbol("a", h1)
        id = TrivialKet
        self.assertEqual(a * id, a)
        self.assertEqual(id * a, a)

    def testOrdering(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        self.assertEqual(a * b,TensorKet(a,b))
        self.assertEqual(a * b, b * a)


    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a * b).space, h1*h2)


    def testEquality(self):
        h1 = local_space("h1")
        h2 = local_space("h2")

        self.assertEqual(CoherentStateKet(h1, 1)*CoherentStateKet(h2, 2), CoherentStateKet(h2, 2) * CoherentStateKet(h1, 1))


class TestScalarTimesKet(unittest.TestCase):
    def testZeroOne(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        z = KetZero

        self.assertEqual(a+a,2*a)
        self.assertEqual(a*1,a)
        self.assertEqual(1*a, a)
        self.assertEqual(a*5,ScalarTimesKet(5, a))
        self.assertEqual(5*a,a*5)
        self.assertEqual(2*a*3, 6*a)
        self.assertEqual(a*5*b, ScalarTimesKet(5, a*b))
        self.assertEqual(a*(5*b), ScalarTimesKet(5, a*b))

        self.assertEqual(0 * a, z)
        self.assertEqual(a * 0, z)
        self.assertEqual(10 * z, z)


    def testScalarCombination(self):
        a = KetSymbol("a", "h1")
        self.assertEqual(a+a, 2*a)
        self.assertEqual(3 * a + 4 * a, 7 * a)
        self.assertEqual(CoherentStateKet(1, "1") + CoherentStateKet(1, "1"), 2 * CoherentStateKet(1, "1"))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        self.assertEqual((5*(a * b)).space, h1*h2)


class TestOperatorTimesKet(unittest.TestCase):

    def testZeroOne(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        A = OperatorSymbol("A", h1)
        Ap = OperatorSymbol("Ap", h1)
        B = OperatorSymbol("B", h2)

        self.assertEqual(IdentityOperator*a, a)
        self.assertEqual(A * (Ap * a), (A * Ap) * a)
        self.assertEqual((A * B) * (a * b), (A * a) * (B * b))



    def testScalarCombination(self):
        a = KetSymbol("a", "h1")
        self.assertEqual(a+a, 2*a)
        self.assertEqual(3 * a + 4 * a, 7 * a)
        self.assertEqual(CoherentStateKet(1, "1") + CoherentStateKet(1, "1"), 2 * CoherentStateKet(1, "1"))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = KetSymbol("a", h1)
        b = KetSymbol("b", h2)
        self.assertEqual((5*(a * b)).space, h1*h2)



class TestLocalOperatorKetRelations(unittest.TestCase):

    def testCreateDestroy(self):
        self.assertEqual(Create(1) * BasisKet(1, 2), sqrt(3) * BasisKet(1, 3))
        self.assertEqual(Destroy(1) * BasisKet(1, 2), sqrt(2) * BasisKet(1, 1))
        self.assertEqual(Destroy(1) * BasisKet(1, 0), KetZero)
        self.assertEqual(Destroy(1) * CoherentStateKet(1, 10.), 10 * CoherentStateKet(1, 10.))


    def testPhase(self):
        self.assertEqual(Phase(1, 5) * BasisKet(1, 3), exp(I * 15) * BasisKet(1, 3))
        self.assertEqual(Phase(1, pi) * CoherentStateKet(1, 3.), CoherentStateKet(1, -3.))

    def testDisplace(self):
        self.assertEqual(Displace(1, 5 + 6j) * CoherentStateKet(1, 3.), exp(I * ((5+6j)*3).imag) * CoherentStateKet(1, 8 + 6j))
        self.assertEqual(Displace(1, 5 + 6j) * BasisKet(1,0), CoherentStateKet(1, 5+6j))

    def testLocalSigmaPi(self):
        self.assertEqual(LocalSigma(1, 0, 1) * BasisKet(1, 1), BasisKet(1, 0))
        self.assertEqual(LocalSigma(1, 0, 0) * BasisKet(1, 1), KetZero)

    def testActLocally(self):
        self.assertEqual((Create(1) * Destroy(2)) * (BasisKet(1, 2) * BasisKet(2, 1)), sqrt(3) * BasisKet(1, 3) * BasisKet(2,0))
