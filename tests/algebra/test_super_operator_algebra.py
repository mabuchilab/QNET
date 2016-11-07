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
from sympy import symbols, sqrt, I, conjugate


from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.super_operator_algebra import (
        SuperOperator, SuperOperatorSymbol, IdentitySuperOperator,
        SuperOperatorPlus, SuperOperatorTimes, SuperOperatorTimesOperator,
        ScalarTimesSuperOperator, ZeroSuperOperator, SPre, SPost, liouvillian,
        liouvillian_normal_form)
from qnet.algebra.pattern_matching import wc, ProtoExpr, pattern_head
from qnet.algebra.operator_algebra import (
        OperatorPlus, ScalarTimesOperator, OperatorSymbol, ZeroOperator,
        Create, Destroy, OperatorTimes)


class TestSuperOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        assert SuperOperatorSymbol("a", 1) == SuperOperatorSymbol("a", 1)

    def testMatch(self):

        A = wc("A", head=SuperOperator)
        a = SuperOperatorSymbol("a", "hs")
        b = SuperOperatorSymbol("b","hs")
        b2 = SuperOperatorSymbol("b","hs")

        assert b == b2
        assert A.match(a)
        assert A.match(a)['A'] == a
        assert A.match(b)
        assert A.match(b)['A'] == b

        expr = ProtoExpr(args=[b, b], kwargs={})
        pat = pattern_head(A, A)
        assert pat.match(expr)
        assert pat.match(expr)['A'] == b

        expr = ProtoExpr(args=[b, b2], kwargs={})
        pat = pattern_head(A, A)
        assert pat.match(expr)
        assert pat.match(expr)['A'] == b


class TestSuperOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = LocalSpace("hs")
        a = SuperOperatorSymbol("a", hs)
        id = IdentitySuperOperator
        assert a+0 == a
        assert 0+a == a
        assert 1 + a + 1 == a + 2
        assert a + 2 == SuperOperatorPlus(ScalarTimesSuperOperator(2,id),a)


    def testAdditionToSuperOperator(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        assert a + b == b + a
        assert a + b == SuperOperatorPlus(a,b)
        assert (a+b).space == h1 * h2

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = SuperOperatorSymbol("a", hs)
        b = SuperOperatorSymbol("b", hs)
        z = ZeroSuperOperator
        assert a-a == z
        assert a-b == SuperOperatorPlus(a, ScalarTimesSuperOperator(-1,b))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        assert (a+b).space == h1 * h2


    def testCommutativity(self):
        h1 = LocalSpace("h1")
        assert SuperOperatorSymbol("A", h1) + SuperOperatorSymbol("B", h1) == \
                            SuperOperatorSymbol("B", h1) + SuperOperatorSymbol("A", h1)




class TestSuperOperatorTimes(unittest.TestCase):

    def testIdentity(self):
        h1 = LocalSpace("h1")
        a = SuperOperatorSymbol("a", h1)
        id = IdentitySuperOperator
        assert a * id == a
        assert id * a == a

    def testOrdering(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        c = SuperOperatorSymbol("c", h2)
        dpre = SPre(SuperOperatorSymbol("d", h1))
        epre = SPre(SuperOperatorSymbol("e", h1))
        dpost = SPost(SuperOperatorSymbol("d", h1))
        epost = SPost(SuperOperatorSymbol("e", h1))

        assert a * b == SuperOperatorTimes(a,b)
        assert b * a == a * b
        assert c * a * b * c * a == SuperOperatorTimes(a, a, c, b, c)

    def testSPreSPostRules(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        d = OperatorSymbol("d", h1)
        e = OperatorSymbol("e", h1)
        dpre = SPre(d)
        epre = SPre(e)
        dpost = SPost(d)
        epost = SPost(e)
        assert dpre * epre == SPre(d * e)
        assert dpost * epost == SPost(e * d)
        assert dpost * epre == SPre(e) * SPost(d)

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        assert a.space == h1
        assert (a * b).space == h1*h2


    def testCommutativity(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        assert a*b == b*a




class TestScalarTimesSuperOperator(unittest.TestCase):

    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b =  SuperOperatorSymbol("b", h2)
        z = ZeroSuperOperator

        assert a+a == 2*a
        assert a*1 == a
        assert 1*a == a
        assert a*5 == ScalarTimesSuperOperator(5, a)
        assert 5*a == a*5
        assert 2*a*3 == 6*a
        assert a*5*b == ScalarTimesSuperOperator(5, a*b)
        assert a*(5*b) == ScalarTimesSuperOperator(5, a*b)

        assert 0 * a == z
        assert a*0 == z
        assert 10 * z == z

    def testEquality(self):
        h1 = LocalSpace("h1")
        a = SuperOperatorSymbol("a", h1)
        assert 5*a == ScalarTimesSuperOperator(5, a)

    def testScalarCombination(self):
        a = SuperOperatorSymbol("a", "h1")
        assert a+a == 2*a
        assert 3*a + 4*a == 7 * a

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        assert (5*(a * b)).space == h1*h2


class TestSuperOperatorTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        B = SuperOperatorSymbol("B", h2)
        z = ZeroSuperOperator
        one = IdentitySuperOperator

        assert one * a == a
        assert z * a == ZeroOperator
        # self.assertEqual(B * a, a * (B * IdentityOperator))

    def testEqual2(self):
        h1 = LocalSpace("h1")
        A = SuperOperatorSymbol("A", h1)
        a = OperatorSymbol("a", h1)

        OTO = SuperOperatorTimesOperator(A, a)
        assert A * a == OTO

    def testCombination(self):

        h1 = LocalSpace("h1")
        a = OperatorSymbol("a", h1)
        A = SuperOperatorSymbol("A", h1)
        B = SuperOperatorSymbol("B", h1)
        assert A * (B * a) == (A * B) * a



    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", h1)
        b = SuperOperatorSymbol("b", h2)
        assert (5 * (a * b)).space == h1 * h2


def test_liouvillian_normal_form():
    kappa_1, kappa_2 = symbols('kappa_1, kappa_2', positive=True)
    Delta = symbols('Delta', real=True)
    alpha = symbols('alpha')
    H = (Delta * Create(1) * Destroy(1) +
         (sqrt(kappa_1) / (2 * I)) *
         (alpha * Create(1) - alpha.conjugate() * Destroy(1)))
    Ls = [sqrt(kappa_1) * Destroy(1) + alpha, sqrt(kappa_2) * Destroy(1)]
    LL = liouvillian(H, Ls)
    Hnf, Lsnf = liouvillian_normal_form(LL)
    Hnf_expected = OperatorPlus(
            ScalarTimesOperator(-I*alpha*sqrt(kappa_1),
                                Create(LocalSpace('1'))),
            ScalarTimesOperator(I*sqrt(kappa_1)*conjugate(alpha),
                                Destroy(LocalSpace('1'))),
            ScalarTimesOperator(Delta, OperatorTimes(Create(LocalSpace('1')),
                                Destroy(LocalSpace('1')))))
    assert Hnf == Hnf_expected
    Lsnf_expected = [ScalarTimesOperator(sqrt(kappa_1 + kappa_2),
                                         Destroy(LocalSpace('1')))]
    assert Lsnf == Lsnf_expected
