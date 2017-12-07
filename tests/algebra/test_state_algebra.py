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
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################


import unittest

from sympy import sqrt, exp, I, pi, Idx, IndexedBase, symbols

from qnet.algebra.abstract_algebra import no_rules, simplify
from qnet.algebra.operator_algebra import (
        OperatorSymbol, Create, Destroy, Jplus, Jminus, Jz, Phase, Displace,
        LocalSigma, IdentityOperator, OperatorPlus)
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.state_algebra import (
        KetSymbol, ZeroKet, KetPlus, ScalarTimesKet, CoherentStateKet,
        TrivialKet, UnequalSpaces, TensorKet, LocalKet, BasisKet, KetBra)
from qnet.algebra.indices import (
    FockIndex, IntIndex, StrLabel, SymbolicLabelBase)
from qnet.algebra.pattern_matching import wc
import pytest


class TestStateAddition(unittest.TestCase):

    def testAdditionToZero(self):
        hs = LocalSpace("hs")
        a = KetSymbol("a", hs=hs)
        z = ZeroKet
        assert a+z == a
        assert z+a == a
        assert z+z == z
        assert z == 0


    def testAdditionToOperator(self):
        hs = LocalSpace("hs")
        a = KetSymbol("a", hs=hs)
        b = KetSymbol("b", hs=hs)
        assert a + b == b + a
        assert a + b == KetPlus(a,b)

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = KetSymbol("a", hs=hs)
        b = KetSymbol("b", hs=hs)
        z = ZeroKet
        lhs = a - a
        assert lhs == z
        lhs = a - b
        rhs = KetPlus(a, ScalarTimesKet(-1,b))
        assert lhs == rhs

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        with pytest.raises(UnequalSpaces):
            a.__add__(b)


    def testEquality(self):
        h1 = LocalSpace("h1")
        assert (CoherentStateKet(10., hs=h1) + CoherentStateKet(20., hs=h1) ==
                CoherentStateKet(20., hs=h1) + CoherentStateKet(10., hs=h1))


class TestTensorKet(unittest.TestCase):

    def testIdentity(self):
        h1 = LocalSpace("h1")
        a = KetSymbol("a", hs=h1)
        id = TrivialKet
        assert a * id == a
        assert id * a == a

    def testOrdering(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        assert a * b == TensorKet(a,b)
        assert a * b == b * a


    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        assert a.space == h1
        assert (a * b).space == h1*h2


    def testEquality(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")

        assert (CoherentStateKet(1, hs=h1) * CoherentStateKet(2, hs=h2) ==
                CoherentStateKet(2, hs=h2) * CoherentStateKet(1, hs=h1))


class TestScalarTimesKet(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        z = ZeroKet

        assert a+a == 2*a
        assert a*1 == a
        assert 1*a == a
        assert a*5 == ScalarTimesKet(5, a)
        assert 5*a == a*5
        assert 2*a*3 == 6*a
        assert a*5*b == ScalarTimesKet(5, a*b)
        assert a*(5*b) == ScalarTimesKet(5, a*b)

        assert 0 * a == z
        assert a * 0 == z
        assert 10 * z == z


    def testScalarCombination(self):
        a = KetSymbol("a", hs="h1")
        assert a+a == 2*a
        assert 3 * a + 4 * a == 7 * a
        assert (CoherentStateKet("1", hs=1) + CoherentStateKet("1", hs=1) ==
                2 * CoherentStateKet("1", hs=1))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        assert (5*(a * b)).space == h1*h2


class TestOperatorTimesKet(unittest.TestCase):

    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        A = OperatorSymbol("A", hs=h1)
        Ap = OperatorSymbol("Ap", hs=h1)
        B = OperatorSymbol("B", hs=h2)

        assert IdentityOperator*a == a
        assert A * (Ap * a) == (A * Ap) * a
        assert (A * B) * (a * b) == (A * a) * (B * b)



    def testScalarCombination(self):
        a = KetSymbol("a", hs="h1")
        assert a+a == 2*a
        assert 3 * a + 4 * a == 7 * a
        assert (CoherentStateKet("1", hs=1) + CoherentStateKet("1", hs=1) ==
                2 * CoherentStateKet("1", hs=1))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = KetSymbol("a", hs=h1)
        b = KetSymbol("b", hs=h2)
        assert (5*(a * b)).space == h1*h2


class TestLocalOperatorKetRelations(unittest.TestCase):

    def testCreateDestroy(self):
        assert Create(hs=1) * BasisKet(2, hs=1) == sqrt(3) * BasisKet(3, hs=1)
        assert Destroy(hs=1) * BasisKet(2, hs=1) == sqrt(2) * BasisKet(1, hs=1)
        assert Destroy(hs=1) * BasisKet(0, hs=1) == ZeroKet
        lhs = Destroy(hs=1) * CoherentStateKet(10., hs=1)
        rhs = 10 * CoherentStateKet(10., hs=1)
        assert lhs == rhs

    def testSpin(self):
        j = 3
        h = LocalSpace("j", basis=range(-j,j+1))

        assert (Jplus(hs=h) * BasisKet('2', hs=h) ==
                sqrt(j*(j+1)-2*(2+1)) * BasisKet('3', hs=h))
        assert (Jminus(hs=h) * BasisKet('2', hs=h) ==
                sqrt(j*(j+1)-2*(2-1)) * BasisKet('1', hs=h))
        assert Jz(hs=h) * BasisKet('2', hs=h) == 2 * BasisKet('2', hs=h)


    def testPhase(self):
        assert (Phase(5, hs=1) * BasisKet(3, hs=1) ==
                exp(I * 15) * BasisKet(3, hs=1))
        lhs = Phase(pi, hs=1) * CoherentStateKet(3., hs=1)
        rhs = CoherentStateKet(-3., hs=1)
        assert lhs.__class__ == rhs.__class__
        assert lhs.space == rhs.space
        assert abs(lhs.ampl - rhs.ampl) < 1e-14

    def testDisplace(self):
        assert (Displace(5 + 6j, hs=1) * CoherentStateKet(3., hs=1) ==
                exp(I * ((5+6j)*3).imag) * CoherentStateKet(8 + 6j, hs=1))
        assert (Displace(5 + 6j, hs=1) * BasisKet(0, hs=1) ==
                CoherentStateKet(5+6j, hs=1))

    def testLocalSigmaPi(self):
        assert (LocalSigma(0, 1, hs = 1) * BasisKet(1, hs=1) ==
                BasisKet(0, hs=1))
        assert (LocalSigma(0, 0, hs = 1) * BasisKet(1, hs=1) ==
                ZeroKet)

    def testActLocally(self):
        assert ((Create(hs=1) * Destroy(hs=2)) *
                (BasisKet(2, hs=1) * BasisKet(1, hs=2)) ==
                sqrt(3) * BasisKet(3, hs=1) * BasisKet(0, hs=2))


    def testOperatorTensorProduct(self):
        assert ((Create(hs=1)*Destroy(hs=2)) *
                (BasisKet(0, hs=1) * BasisKet(1, hs=2)) ==
                BasisKet(1, hs=1) * BasisKet(0, hs=2))

    def testOperatorProduct(self):
        assert ((Create(hs=1) * Destroy(hs=1)) *
                (BasisKet(1, hs=1) * BasisKet(1, hs=2)) ==
                BasisKet(1, hs=1) * BasisKet(1, hs=2))
        assert ((Create(hs=1) * Destroy(hs=1) * Destroy(hs=1)) *
                (BasisKet(2, hs=1)*BasisKet(1, hs=2)) ==
                sqrt(2) * BasisKet(1, hs=1) * BasisKet(1, hs=2))
        assert ((Create(hs=1) * Destroy(hs=1) * Destroy(hs=1)) *
                BasisKet(2, hs=1) ==
                sqrt(2) * BasisKet(1, hs=1))
        assert ((Create(hs=1) * Destroy(hs=1)) * BasisKet(1, hs=1) ==
                BasisKet(1, hs=1))
        assert ((Create(hs=1) * Destroy(hs=1)) * BasisKet(0, hs=1) == ZeroKet)


def test_expand_ketbra():
    """Test expansion of KetBra"""
    hs = LocalSpace('0', basis=('0', '1'))
    expr = KetBra(
        KetPlus(BasisKet('0', hs=hs), BasisKet('1', hs=hs)),
        KetPlus(BasisKet('0', hs=hs), BasisKet('1', hs=hs)))
    with no_rules(KetBra):
        expr_expand = expr.expand()
    assert expr_expand == OperatorPlus(
        KetBra(BasisKet('0', hs=hs), BasisKet('0', hs=hs)),
        KetBra(BasisKet('0', hs=hs), BasisKet('1', hs=hs)),
        KetBra(BasisKet('1', hs=hs), BasisKet('0', hs=hs)),
        KetBra(BasisKet('1', hs=hs), BasisKet('1', hs=hs)))


def eval_lb(expr, mapping):
    """Evaluate symbolic labels with the given mapping"""
    return simplify(expr, rules=[(
        wc('label', head=SymbolicLabelBase),
        lambda label: label.evaluate(mapping))])


def test_ket_symbolic_labels():
    """Test that we can instantiate Kets with symbolic labels"""
    i = Idx('i')
    i_sym = symbols('i')
    j = Idx('j')
    hs0 = LocalSpace(0)
    hs1 = LocalSpace(1)
    Psi = IndexedBase('Psi')

    assert (
        eval_lb(BasisKet(FockIndex(2 * i), hs=hs0), {i: 2}) ==
        BasisKet(4, hs=hs0))
    assert (
        eval_lb(BasisKet(FockIndex(2 * i_sym), hs=hs0), {i_sym: 2}) ==
        BasisKet(4, hs=hs0))
    with pytest.raises(TypeError) as exc_info:
        BasisKet(IntIndex(2 * i), hs=hs0)
    assert "not IntIndex" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        BasisKet(StrLabel(2 * i), hs=hs0)
    assert "not StrLabel" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        BasisKet(2 * i, hs=hs0)
    assert "not Mul" in str(exc_info.value)

    assert(
        eval_lb(LocalKet(StrLabel(2 * i), hs=hs0), {i: 2}) ==
        LocalKet("4", hs=hs0))
    with pytest.raises(TypeError) as exc_info:
        eval_lb(LocalKet(FockIndex(2 * i), hs=hs0), {i: 2})
    assert "type of label must be str" in str(exc_info.value)

    assert StrLabel(Psi[i, j]).evaluate({i: 'i', j: 'j'}) == 'Psi_i,j'
    assert(
        eval_lb(
            KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1), {i: 'i', j: 'j'}) ==
        KetSymbol("Psi_i,j", hs=hs0*hs1))
    assert(
        eval_lb(
            KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1), {i: 1, j: 2}) ==
        KetSymbol("Psi_1,2", hs=hs0*hs1))

    assert (
        eval_lb(
            LocalSigma(FockIndex(i), FockIndex(j), hs=hs0), {i: 1, j: 2}) ==
        LocalSigma(1, 2, hs=hs0))
    assert (
        BasisKet(FockIndex(i), hs=hs0) * BasisKet(FockIndex(j), hs=hs0).dag ==
        LocalSigma(FockIndex(i), FockIndex(j), hs=hs0))
