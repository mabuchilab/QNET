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

from numpy import (array as np_array, conjugate as np_conjugate,
                   int_ as np_int, float_ as np_float)
from sympy import symbols, sqrt, I, exp, sympify

from qnet.algebra.operator_algebra import (
        Displace, Create, Destroy, OperatorSymbol, IdentityOperator,
        ZeroOperator, OperatorPlus, LocalSigma, LocalProjector, OperatorTrace,
        Adjoint, X, Y, Z, ScalarTimesOperator, OperatorTimes, Jz,
        Jplus, Jminus, Phase)
from qnet.algebra.matrix_algebra import Matrix, identity_matrix
from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, TrivialSpace, ProductSpace)


def test_identity_singleton():
    """Test the Singleton properties of the IdentityOperator"""
    assert IdentityOperator() is IdentityOperator
    assert IdentityOperator.__class__() is IdentityOperator
    assert IdentityOperator.__class__.create() is IdentityOperator
    assert IdentityOperator.create() is IdentityOperator
    assert IdentityOperator.create(
        *IdentityOperator.args, **IdentityOperator.kwargs) is IdentityOperator


def test_identity_comparisons():
    assert IdentityOperator == 1
    assert IdentityOperator == np_float(1.0)
    assert IdentityOperator == sympify(1)

    assert IdentityOperator != np_int(-3)
    assert IdentityOperator != 0.0
    assert IdentityOperator != sympify(3.5)


def test_zero_comparisons():
    assert ZeroOperator == np_int(0)
    assert ZeroOperator == 0.0
    assert ZeroOperator == sympify(0.0)

    assert ZeroOperator != -3
    assert ZeroOperator != np_float(1.0)
    assert ZeroOperator != sympify(2)


class TestOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        assert Create(hs="1") == Create(hs="1")
        assert OperatorSymbol("a", hs=1) == OperatorSymbol("a", hs=1)

    def testImplicitHilbertSpaceCreation(self):
        hs = LocalSpace("hs")
        h2 = LocalSpace("h2")
        aa = OperatorSymbol("aa", hs=hs)
        bb = OperatorSymbol("aa", hs=hs*h2)
        a = Destroy(hs=hs)
        assert aa == OperatorSymbol("aa", hs="hs")
        assert bb == OperatorSymbol("aa", hs=("hs", "h2"))
        assert a == Destroy(hs="hs")
        assert Destroy(hs=1) == Destroy(hs="1")
        assert OperatorSymbol("a", hs=1) == OperatorSymbol("a", hs="1")


class TestOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs=hs)
        id_ = IdentityOperator
        assert a+0 == a
        assert 0+a == a
        assert 1 + a + 1 == a + 2
        lhs = a + 2
        rhs = OperatorPlus(ScalarTimesOperator(2, id_), a)
        assert lhs == rhs

    def testOperatorOrdering(self):
        hs = LocalSpace("1")
        a = OperatorSymbol("a", hs=hs)
        b = OperatorSymbol("b", hs=hs)
        c = OperatorSymbol("c", hs=hs)
        assert c+b+a == OperatorPlus(a, b, c)

    def testAdditionToOperator(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs=hs)
        b = OperatorSymbol("b", hs=hs)
        assert a + b == b + a
        assert a + b == OperatorPlus(a, b)

    def testAdditionToOperatorProduct(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs=hs)
        b = OperatorSymbol("b", hs=hs)
        assert a + b*b*a == b*b*a + a
        assert a + b*b*a == OperatorPlus(a, OperatorTimes(b, b, a))

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs=hs)
        b = OperatorSymbol("b", hs=hs)
        z = ZeroOperator
        assert a-a == z
        assert a-b == OperatorPlus(a, ScalarTimesOperator(-1, b))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", hs=h1)
        b = OperatorSymbol("b", hs=h2)
        assert a.space == h1
        assert (a + b).space == h1*h2

    def testEquality(self):
        c1 = Create(hs=1)
        c2 = Create(hs=2)
        assert c1 + c2 == c2 + c1


class TestOperatorTimes(unittest.TestCase):
    def testIdentity(self):
        h1 = LocalSpace("h1")
        a = OperatorSymbol("a", hs=h1)
        id_ = IdentityOperator
        assert a * id_ == a
        assert id_ * a == a

    def testOrdering(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", hs=h1)
        b = OperatorSymbol("b", hs=h2)
        c = OperatorSymbol("c", hs=h2)
        assert a * b == OperatorTimes(a,b)

        assert b * a == a * b
        assert c * a * b * c * a == OperatorTimes(a, a, c, b, c)


    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", hs=h1)
        b = OperatorSymbol("b", hs=h2)
        assert a.space == h1
        assert (a * b).space == h1*h2


    def testEquality(self):
        assert Create(hs=1) * Create(hs=2) == Create(hs=1) * Create(hs=2)


class TestScalarTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", hs=h1)
        b = OperatorSymbol("b", hs=h2)
        z = ZeroOperator

        assert a+a == 2*a
        assert a*1 == a
        assert 1*a == a
        assert a*5 == ScalarTimesOperator(5, a)
        assert 5*a == a*5
        assert 2*a*3 == 6*a
        assert a*5*b == ScalarTimesOperator(5, a*b)
        assert a*(5*b) == ScalarTimesOperator(5, a*b)

        assert 0 * a == z
        assert a*0 == z
        assert 10 * z == z

    def testScalarTimesIdentity(self):
        id_ = IdentityOperator

        assert 2 * id_ == 2
        assert id_ * 2 == np_float(2.0)
        assert -3.7 * id_ == sympify(-3.7)


    def testHashability(self):
        assert hash(ScalarTimesOperator(1, Create(hs=1))) == \
               hash(ScalarTimesOperator(1, Create(hs=1)))

    def testEquality(self):
        assert 5 * Create(hs=1) == (6-1) * Create(hs=1)

    def testScalarCombination(self):
        a = OperatorSymbol("a", hs="h1")
        assert a+a == 2*a
        assert 3*a + 4*a == 7 * a
        assert Create(hs="1") + Create(hs="1") == 2 * Create(hs="1")

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", hs=h1)
        b = OperatorSymbol("b", hs=h2)
        assert (5*(a * b)).space == h1*h2


class TestDifferentiation(unittest.TestCase):

    def testConstantOps(self):
        x = symbols("x")

        X = OperatorSymbol("X", hs=1)
        assert X.diff(x) == ZeroOperator
        assert (2*X).diff(x) == ZeroOperator
        assert X.dag().diff(x) == ZeroOperator

        a = Destroy(hs=1)
        assert a.diff(x) == ZeroOperator
        assert a.dag().diff(x) == ZeroOperator
        assert (a + a.dag()).diff(x) == ZeroOperator
        assert (a * a.dag()).diff(x) == ZeroOperator

        s = LocalSigma(1, 2, hs=1)
        assert s.diff(x) == ZeroOperator

    def testNonConstantOps(self):
        x = symbols("x", real=True)

        X = OperatorSymbol("X", hs=1)
        Y = OperatorSymbol("Y", hs=1)
        assert (x*X).diff(x) == X
        assert ((2*x**2)*X).diff(x) == 4*x*X
        assert (x*X).dag().diff(x) == X.dag()
        assert (x*X + X).diff(x) == X

        assert (x*X + (x**2)*Y).diff(x) == X + 2*x*Y
        assert (x*X + (x**2)*Y).diff(x, 2) == 2*Y

        assert ((x*X) * (x**2)*Y).diff(x) == 3*x**2 * X * Y
        assert ((x*X + Y) * (x**2)*Y).diff(x) == 3*x**2 * X * Y + 2*x*Y*Y


class TestLocalOperatorRelations(unittest.TestCase):
    def testCommutatorAAdag(self):
        h = LocalSpace("h")
        ii = IdentityOperator
        dc = Destroy(hs=h) * Create(hs=h)
        cd = Create(hs=h) * Destroy(hs=h)
        lhs = dc - cd
        assert lhs == ii



    def testSpin(self):
        j = 3
        h = LocalSpace("h", basis=range(-j,j+1))
        jz = Jz(hs=h)
        jp = Jplus(hs=h)
        jm = Jminus(hs=h)

        assert (jp*jm-jm*jp).expand() == 2*jz
        assert (jz*jm-jm*jz).expand() == -jm
        assert (jz*jp-jp*jz).expand() == jp

        assert jp*LocalProjector('3', hs=h) == ZeroOperator
        assert (jp*LocalProjector('2', hs=h) ==
                sqrt(j*(j+1)-2*(2+1)) * LocalSigma('3', '2', hs=h))

        assert jm*LocalProjector('-3', hs=h) == ZeroOperator
        assert (jm*LocalProjector('-2', hs=h) ==
                sqrt(j*(j+1)-2*(2+1)) * LocalSigma('-3', '-2', hs=h))

        assert jz*LocalProjector('-3', hs=h) == -3*LocalProjector('-3', hs=h)

        assert LocalProjector('3', hs=h)*jm == ZeroOperator
        assert (LocalProjector('2', hs=h)*jm ==
                sqrt(j*(j+1)-2*(2+1))*LocalSigma('2', '3', hs=h))

        assert LocalProjector('-3', hs=h)*jp == ZeroOperator
        assert (LocalProjector('-2', hs=h)*jp ==
                sqrt(j*(j+1)-2*(2+1))*LocalSigma('-2', '-3', hs=h))

        assert LocalProjector('-3', hs=h)*jz == -3*LocalProjector('-3', hs=h)

    def testPhase(self):
        assert Phase(5, hs=1).adjoint() == Phase(-5, hs=1)
        assert Phase(5, hs=1) * Phase(-5, hs=1) == IdentityOperator
        assert (Phase(5, hs=1) * Create(hs=1) * Phase(-5, hs=1) ==
                exp(I * 5) * Create(hs=1))
        assert (Phase(5, hs=1) * LocalSigma(3, 4, hs=1) ==
                exp(15 * I) * LocalSigma(3,4, hs=1))
        assert (LocalSigma(3,4, hs=1) * Phase(5, hs=1) ==
                exp(20 * I) * LocalSigma(3,4, hs=1))
        assert Phase(5, hs=1) * LocalSigma(0,4, hs=1) == LocalSigma(0,4, hs=1)
        assert LocalSigma(3,0, hs=1) * Phase(5, hs=1) == LocalSigma(3,0, hs=1)

    def testDisplace(self):
        assert Displace(5+6j, hs=1).adjoint() == Displace(-5-6j, hs=1)
        assert (Displace(5+6j, hs=1) * Displace(-5-6j, hs=1) ==
                IdentityOperator)
        assert (Displace(5+6j, hs=1) * Create(hs=1) * Displace(-5-6j, hs=1) ==
                Create(hs=1) - (5-6j))

    def testLocalSigmaPi(self):
        h = LocalSpace("h")
        assert (LocalSigma(0, 1, hs=h) * LocalSigma(1, 2, hs=h) ==
                LocalSigma(0, 2, hs=h))
        assert LocalSigma(0, 0, hs=h) == LocalProjector(0, hs=h)

    def testAnnihilation(self):
        h = LocalSpace("h")
        z = ZeroOperator
        assert Destroy(hs=h) * LocalSigma(0, 1, hs=h) == z
        assert LocalSigma(1, 0, hs=h) * Create(hs=h) == z


class TestOperatorTrace(unittest.TestCase):

    def testConstruction(self):
        M = OperatorSymbol("M", hs=1)
        N = OperatorSymbol("N", hs=ProductSpace(LocalSpace(1), LocalSpace(2)))
        assert (OperatorTrace.create(M, over_space=1) ==
                OperatorTrace(M, over_space=1))
        assert OperatorTrace(M, over_space=1).space == TrivialSpace
        assert OperatorTrace(N, over_space=1).space == LocalSpace(2)

    def testSimplificationPlus(self):
        M = OperatorSymbol("M", hs=1)
        N = OperatorSymbol("N", hs=1)
        O = OperatorSymbol("O", hs=1)

        assert (OperatorTrace.create(M+N, over_space=1) ==
                (OperatorTrace(M, over_space=1) +
                 OperatorTrace(N, over_space=1)))
        assert (OperatorTrace.create((M+N)*O, over_space=1).expand() ==
                (OperatorTrace(M*O, over_space=1) +
                 OperatorTrace(N*O, over_space=1)))

    def testSimplificationTimes(self):
        M = OperatorSymbol("M", hs=1)
        N = OperatorSymbol("N", hs=2)
        O = OperatorSymbol("O", hs=ProductSpace(LocalSpace(1), LocalSpace(2),
                                                LocalSpace(3)))
        assert (OperatorTrace.create(M * N, over_space=1) ==
                OperatorTrace(M, over_space=1) * N)
        lhs = OperatorTrace.create(
                     M*N*O,
                     over_space=ProductSpace(LocalSpace(2), LocalSpace(3)))
        rhs = M * OperatorTrace(N * OperatorTrace(O, over_space=3),
                                over_space=2)
        assert lhs == rhs
        assert (OperatorTrace.create(
                    OperatorTrace.create(N, over_space=2) * M,
                    over_space=1
                ) == (
                    OperatorTrace(M, over_space=1) *
                    OperatorTrace(N, over_space=2)
                ))
        assert (OperatorTrace.create(
                    M * N,
                    over_space=ProductSpace(LocalSpace(1),LocalSpace(2))
                ) == (
                    (OperatorTrace(M, over_space=1) *
                     OperatorTrace(N, over_space=2))
                ))

    def testSimplificationScalarTimesOperator(self):
        M = OperatorSymbol("M", hs=1)
        assert (OperatorTrace.create(10 * M, over_space=1) ==
                10 * OperatorTrace(M, over_space=1))

    def testSimplificationAdjoint(self):
        M = OperatorSymbol("M", hs=1)
        assert (OperatorTrace.create(M.adjoint(), over_space=1) ==
                Adjoint(OperatorTrace(M, over_space=1)))

    def testLocalOps(self):
        op = OperatorTrace.create(Create(hs=1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(Destroy(hs=1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma(1, 2, hs=1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma(1, 1, hs=1), over_space=1)
        assert op == IdentityOperator
        hs = LocalSpace(1, basis=('g', 'e'))
        op = OperatorTrace.create(LocalSigma('e', 'g', hs=hs), over_space=hs)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma('e', 'e', hs=hs), over_space=hs)
        assert op == IdentityOperator

    def testSimplificationMaxwellBloch(self):

        a = LocalSpace("a", basis=('h','g'))
        f = LocalSpace("f")
        x,y,z = symbols("x,y,z", real = True)
        alpha = symbols("alpha")
        rho_a = (IdentityOperator + x * X(a) + y * Y(a) + z * Z(a)) / 2
        sigma = X(a) + I*Y(a)
        rho_f = (Displace(alpha, hs=f) * LocalProjector(0, hs=f) *
                 Displace(-alpha, hs=f))
        rho = rho_a * rho_f
        lhs = OperatorTrace.create(rho, over_space=ProductSpace(a, f))
        lhs = lhs.expand()
        assert lhs == IdentityOperator

    def testDimensionPrefactor(self):
        h1 = LocalSpace(1, dimension=10)
        P = OperatorSymbol("P", hs=2)
        lhs = OperatorTrace.create(P, over_space=h1)
        rhs = 10 * P
        assert lhs == rhs


class TestOperatorMatrices(unittest.TestCase):
    def testConstruction(self):
        h1, h2, h3 = LocalSpace("h1"), LocalSpace("h2"), LocalSpace("h3")
        a, b, c = Destroy(hs=h1), Destroy(hs=h2), Destroy(hs=h3)
        assert np_conjugate(a) == a.dag()

        M = Matrix([[a,b],[c,a]])
#        self.assertEqual(M.matrix, np_array([[a,b],[b,a]]))
        assert M == Matrix(np_array([[a,b],[c,a]]))
        assert M.T == Matrix(np_array([[a,c],[b,a]]))
        assert M.conjugate() == Matrix(np_array([[a.dag(),b.dag()],[c.dag(),a.dag()]]))
        assert M.H == Matrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]]))
        assert M.H == Matrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]]))

    def testMathOperations(self):
        M = Matrix([[Create(hs="1"), 0],[0, Destroy(hs="1")]])
        N = Matrix([[Destroy(hs="1"), Create(hs="2")],[0, Destroy(hs="1")]])
        assert M+N == Matrix([[Create(hs="1")+Destroy(hs="1"), Create(hs="2")],[0, 2*Destroy(hs="1")]])
        assert M*N == Matrix([[Create(hs="1")*Destroy(hs="1"), Create(hs="1")*Create(hs="2")],[0, Destroy(hs="1")*Destroy(hs="1")]])
        assert IdentityOperator * M == M
        assert 1 * M == M
        assert Create(hs="1") * identity_matrix(2) == Matrix([[Create(hs="1"),0],[0,Create(hs="1")]])

    def testElementExpand(self):
        assert Matrix([[(Create(hs=1) + Create(hs=2)) * Create(hs=3)]]).expand() == Matrix([[Create(hs=1)*Create(hs=3) + Create(hs=2)*Create(hs=3)]])
