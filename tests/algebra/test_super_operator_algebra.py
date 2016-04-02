# coding=utf-8
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

from qnet.algebra.super_operator_algebra import *


class TestSuperOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        self.assertEqual(SuperOperatorSymbol("a", 1), SuperOperatorSymbol("a",1))

    def testMatch(self):

        A = wc("A", head = SuperOperator)
        a = SuperOperatorSymbol("a", "hs")
        b = SuperOperatorSymbol("b","hs")
        b2 = SuperOperatorSymbol("b","hs")

        self.assertEqual(b,b2)
        self.assertEqual(match(A, a),Match(A = a))
        self.assertEqual(match(A, b),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((b, b))),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((b, b2))),Match(A = b))



class TestSuperOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = local_space("hs")
        a = SuperOperatorSymbol("a", hs)
        id = IdentitySuperOperator
        self.assertEqual(a+0, a)
        self.assertEqual(0+a, a)
        self.assertEqual(1 + a + 1, a + 2)
        self.assertEqual(a + 2, SuperOperatorPlus(ScalarTimesSuperOperator(2,id),a))


    def testAdditionToSuperOperator(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        self.assertEqual(a + b, b + a)
        self.assertEqual(a + b, SuperOperatorPlus(a,b))
        self.assertEqual((a+b).space, h1 * h2)

    def testSubtraction(self):
        hs = local_space("hs")
        a = SuperOperatorSymbol("a", hs)
        b = SuperOperatorSymbol("b", hs)
        z = ZeroSuperOperator
        self.assertEqual(a-a, z)
        self.assertEqual(a-b, SuperOperatorPlus(a, ScalarTimesSuperOperator(-1,b)))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        self.assertEqual((a+b).space, h1 * h2)


    def testCommutativity(self):
        h1 = local_space("h1")
        self.assertEqual(SuperOperatorSymbol("A", h1) + SuperOperatorSymbol("B", h1),
                            SuperOperatorSymbol("B", h1) + SuperOperatorSymbol("A", h1))




class TestSuperOperatorTimes(unittest.TestCase):

    def testIdentity(self):
        h1 = local_space("h1")
        a = SuperOperatorSymbol("a", h1)
        id = IdentitySuperOperator
        self.assertEqual(a * id, a)
        self.assertEqual(id * a, a)

    def testOrdering(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        c = SuperOperatorSymbol("c", h2)
        dpre = SPre(SuperOperatorSymbol("d", h1))
        epre = SPre(SuperOperatorSymbol("e", h1))
        dpost = SPost(SuperOperatorSymbol("d", h1))
        epost = SPost(SuperOperatorSymbol("e", h1))

        self.assertEqual(a * b, SuperOperatorTimes(a,b))
        self.assertEqual(b * a, a * b)
        self.assertEqual(c * a * b * c * a, SuperOperatorTimes(a, a, c, b, c))

    def testSPreSPostRules(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        d = OperatorSymbol("d", h1)
        e = OperatorSymbol("e", h1)
        dpre = SPre(d)
        epre = SPre(e)
        dpost = SPost(d)
        epost = SPost(e)
        self.assertEqual(dpre * epre, SPre(d * e))
        self.assertEqual(dpost * epost, SPost(e * d))
        self.assertEqual(dpost * epre, SPre(e) * SPost(d))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a * b).space, h1*h2)


    def testCommutativity(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        self.assertEqual(a*b, b*a)




class TestScalarTimesSuperOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b =  SuperOperatorSymbol("b", h2)
        z = ZeroSuperOperator

        self.assertEqual(a+a,2*a)
        self.assertEqual(a*1,a)
        self.assertEqual(1*a, a)
        self.assertEqual(a*5,ScalarTimesSuperOperator(5, a))
        self.assertEqual(5*a,a*5)
        self.assertEqual(2*a*3, 6*a)
        self.assertEqual(a*5*b, ScalarTimesSuperOperator(5, a*b))
        self.assertEqual(a*(5*b), ScalarTimesSuperOperator(5, a*b))

        self.assertEqual(0 * a, z)
        self.assertEqual(a*0, z)
        self.assertEqual(10 * z, z)

    def testEquality(self):
        h1 = local_space("h1")
        a = SuperOperatorSymbol("a", h1)
        self.assertEqual(5*a, ScalarTimesSuperOperator(5, a))

    def testScalarCombination(self):
        a = SuperOperatorSymbol("a", "h1")
        self.assertEqual(a+a, 2*a)
        self.assertEqual(3*a + 4*a, 7 * a)

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        self.assertEqual((5*(a * b)).space, h1*h2)


class TestSuperOperatorTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        B = SuperOperatorSymbol("B", h2)
        z = ZeroSuperOperator
        one = IdentitySuperOperator

        self.assertEqual(one * a, a)
        self.assertEqual(z * a, ZeroOperator)
        # self.assertEqual(B * a, a * (B * IdentityOperator))

    def testEqual2(self):
        h1 = local_space("h1")
        A = SuperOperatorSymbol("A", h1)
        a = OperatorSymbol("a", h1)

        OTO = SuperOperatorTimesOperator(A, a)
        self.assertEqual(A * a, OTO)

    def testCombination(self):

        h1 = local_space("h1")
        a = OperatorSymbol("a", h1)
        A = SuperOperatorSymbol("A", h1)
        B = SuperOperatorSymbol("B", h1)
        self.assertEqual(A * (B * a), (A * B) * a)



    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        self.assertEqual((5 * (a * b)).space, h1 * h2)
