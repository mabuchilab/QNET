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

from numpy import array as np_array, conjugate as np_conjugate
from sympy import symbols, sqrt, I, exp

from qnet.algebra.operator_algebra import (
        Displace, Create, Destroy, OperatorSymbol, IdentityOperator,
        ZeroOperator, OperatorPlus, LocalSigma, LocalProjector, OperatorTrace,
        Adjoint, X, Y, Z, ScalarTimesOperator, OperatorTimes, Jz,
        Jplus, Jminus, Phase)
from qnet.algebra.matrix_algebra import Matrix, identity_matrix
from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, TrivialSpace, ProductSpace)


class TestOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        assert Create("1") == Create("1")
        assert OperatorSymbol("a", 1) == OperatorSymbol("a",1)

    def testImplicitHilbertSpaceCreation(self):
        hs = LocalSpace("hs")
        h2 = LocalSpace("h2")
        aa = OperatorSymbol("aa", hs)
        bb = OperatorSymbol("aa", hs * h2)
        a = Destroy(hs)
        assert aa == OperatorSymbol("aa", "hs")
        assert bb == OperatorSymbol("aa", ("hs", "h2"))
        assert a == Destroy("hs")
        assert Destroy(1) == Destroy("1")
        assert OperatorSymbol("a", 1) == OperatorSymbol("a", "1")


class TestOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        id = IdentityOperator
        assert a+0 == a
        assert 0+a == a
        assert 1 + a + 1 == a + 2
        lhs = a + 2
        rhs = OperatorPlus(ScalarTimesOperator(2, id), a)
        assert lhs == rhs

    def testOperatorOrdering(self):
        hs = LocalSpace("1")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        c = OperatorSymbol("c", hs)
        assert c+b+a == OperatorPlus(a, b, c)

    def testAdditionToOperator(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        assert a + b == b + a
        assert a + b == OperatorPlus(a, b)

    def testAdditionToOperatorProduct(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        assert a + b*b*a == b*b*a + a
        assert a + b*b*a == OperatorPlus(a, OperatorTimes(b, b, a))

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        z = ZeroOperator
        assert a-a == z
        assert a-b == OperatorPlus(a, ScalarTimesOperator(-1, b))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        assert a.space == h1
        assert (a + b).space == h1*h2

    def testEquality(self):
        assert Create(1)+Create(2) == Create(2)+Create(1)


class TestOperatorTimes(unittest.TestCase):
    def testIdentity(self):
        h1 = LocalSpace("h1")
        a = OperatorSymbol("a", h1)
        id = IdentityOperator
        assert a * id == a
        assert id * a == a

    def testOrdering(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        c = OperatorSymbol("c", h2)
        assert a * b == OperatorTimes(a,b)

        assert b * a == a * b
        assert c * a * b * c * a == OperatorTimes(a, a, c, b, c)


    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        assert a.space == h1
        assert (a * b).space == h1*h2


    def testEquality(self):
        assert Create(1)*Create(2) == Create(1)*Create(2)


class TestScalarTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
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

    def testHashability(self):
        assert hash(ScalarTimesOperator(1, Create(1))) == \
               hash(ScalarTimesOperator(1, Create(1)))

    def testEquality(self):
        assert 5 * Create(1) == (6-1) * Create(1)

    def testScalarCombination(self):
        a = OperatorSymbol("a", "h1")
        assert a+a == 2*a
        assert 3*a + 4*a == 7 * a
        assert Create("1") + Create("1") == 2 * Create("1")

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        assert (5*(a * b)).space == h1*h2


class TestDifferentiation(unittest.TestCase):

    def testConstantOps(self):
        x = symbols("x")

        X = OperatorSymbol("X", 1)
        assert X.diff(x) == ZeroOperator
        assert (2*X).diff(x) == ZeroOperator
        assert X.dag().diff(x) == ZeroOperator

        a = Destroy(1)
        assert a.diff(x) == ZeroOperator
        assert a.dag().diff(x) == ZeroOperator
        assert (a + a.dag()).diff(x) == ZeroOperator
        assert (a * a.dag()).diff(x) == ZeroOperator

        s = LocalSigma(1,1,2)
        assert s.diff(x) == ZeroOperator

    def testNonConstantOps(self):
        x = symbols("x", real=True)

        X = OperatorSymbol("X", 1)
        Y = OperatorSymbol("Y", 1)
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
        dc = Destroy(h) * Create(h)
        cd = Create(h) * Destroy(h)
        lhs = dc - cd
        assert lhs == ii



    def testSpin(self):
        j = 3
        h = LocalSpace("h", basis=range(-j,j+1))
        jz = Jz(h)
        jp = Jplus(h)
        jm = Jminus(h)

        assert (jp*jm-jm*jp).expand() == 2*jz
        assert (jz*jm-jm*jz).expand() == -jm
        assert (jz*jp-jp*jz).expand() == jp

        assert jp*LocalProjector(h,3) == ZeroOperator
        assert jp*LocalProjector(h,2) == sqrt(j*(j+1)-2*(2+1))*LocalSigma(h, 3, 2)

        assert jm*LocalProjector(h,-3) == ZeroOperator
        assert jm*LocalProjector(h,-2) == sqrt(j*(j+1)-2*(2+1))*LocalSigma(h, -3, -2)

        assert jz*LocalProjector(h,-3) == -3*LocalProjector(h, -3)

        assert LocalProjector(h,3)*jm == ZeroOperator
        assert LocalProjector(h,2)*jm == sqrt(j*(j+1)-2*(2+1))*LocalSigma(h, 2, 3)

        assert LocalProjector(h,-3)*jp == ZeroOperator
        assert LocalProjector(h,-2)*jp == sqrt(j*(j+1)-2*(2+1))*LocalSigma(h, -2, -3)

        assert LocalProjector(h,-3)*jz == -3*LocalProjector(h, -3)

    def testPhase(self):
        assert Phase(1, 5).adjoint() == Phase(1, -5)
        assert Phase(1, 5) * Phase(1, -5) == IdentityOperator
        assert Phase(1, 5) * Create(1) * Phase(1, -5) == exp(I * 5) * Create(1)
        assert Phase(1, 5) * LocalSigma(1, 3, 4) == exp(15 * I) * LocalSigma(1,3,4)
        assert LocalSigma(1,3,4) * Phase(1, 5) == exp(20 * I) * LocalSigma(1,3,4)
        assert Phase(1, 5) * LocalSigma(1,0,4) == LocalSigma(1,0,4)
        assert LocalSigma(1,3,0) * Phase(1, 5) == LocalSigma(1,3,0)

    def testDisplace(self):
        assert Displace(1, 5 + 6j).adjoint() == Displace(1, -5-6j)
        assert Displace(1, 5+6j) * Displace(1, -5-6j) == IdentityOperator
        assert Displace(1, 5+6j) * Create(1) * Displace(1, -5-6j) == Create(1) - (5-6j)

#    def testSqueeze(self):
#        self.assertEqual(Squeeze(1, 5 + 6j).adjoint(), Squeeze(1, -5-6j))
#        self.assertEqual(Squeeze(1, 5+6j) * Squeeze(1, -5-6j), 1)
#        self.assertEqual(Squeeze(1, 1 + I) * Create(1) * Squeeze(1, -1 - I), cosh(sqrt(2)) * Create(1) - sinh(sqrt(2)) * exp(-I*pi/4)* Destroy(1))

    def testLocalSigmaPi(self):
        h = LocalSpace("h")
        assert LocalSigma(h, 0, 1)* LocalSigma(h, 1, 2) == LocalSigma(h, 0, 2)
        assert LocalSigma(h, 0, 0) == LocalProjector(h, 0)

    def testAnnihilation(self):
        h = LocalSpace("h")
        z = ZeroOperator
        assert Destroy(h) * LocalSigma(h, 0, 1) == z
        assert LocalSigma(h, 1, 0) * Create(h) == z


class TestOperatorTrace(unittest.TestCase):

    def testConstruction(self):
        M = OperatorSymbol("M", 1)
        N = OperatorSymbol("N", ProductSpace(LocalSpace(1), LocalSpace(2)))
        assert (OperatorTrace.create(M, over_space=1) ==
                OperatorTrace(M, over_space=1))
        assert OperatorTrace(M, over_space=1).space == TrivialSpace
        assert OperatorTrace(N, over_space=1).space == LocalSpace(2)

    def testSimplificationPlus(self):
        M = OperatorSymbol("M", 1)
        N = OperatorSymbol("N", 1)
        O = OperatorSymbol("O", 1)

        assert (OperatorTrace.create(M+N, over_space=1) ==
                (OperatorTrace(M, over_space=1) +
                 OperatorTrace(N, over_space=1)))
        assert (OperatorTrace.create((M+N)*O, over_space=1).expand() ==
                (OperatorTrace(M*O, over_space=1) +
                 OperatorTrace(N*O, over_space=1)))

    def testSimplificationTimes(self):
        M = OperatorSymbol("M", 1)
        N = OperatorSymbol("N", 2)
        O = OperatorSymbol("O", ProductSpace(LocalSpace(1), LocalSpace(2),
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
        M = OperatorSymbol("M", 1)
        assert (OperatorTrace.create(10 * M, over_space=1) ==
                10 * OperatorTrace(M, over_space=1))

    def testSimplificationAdjoint(self):
        M = OperatorSymbol("M", 1)
        assert (OperatorTrace.create(M.adjoint(), over_space=1) ==
                Adjoint(OperatorTrace(M, over_space=1)))

    def testLocalOps(self):
        op = OperatorTrace.create(Create(1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(Destroy(1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma(1, 1, 2), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma(1, 1, 1), over_space=1)
        assert op == IdentityOperator
        op = OperatorTrace.create(LocalSigma(1, 'e', 'g'), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma(1, 'e', 'e'), over_space=1)
        assert op == IdentityOperator

    def testSimplificationMaxwellBloch(self):

        a = LocalSpace("a", basis=('h','g'))
        f = LocalSpace("f")
        x,y,z = symbols("x,y,z", real = True)
        alpha = symbols("alpha")
        rho_a = (IdentityOperator + x * X(a) + y * Y(a) + z * Z(a)) / 2
        sigma = X(a) + I*Y(a)
        rho_f = Displace(f, alpha) * LocalProjector(f, 0) * Displace(f, -alpha)
        rho = rho_a * rho_f
        lhs = OperatorTrace.create(rho, over_space=ProductSpace(a, f))
        lhs = lhs.expand()
        assert lhs == IdentityOperator

    def testDimensionPrefactor(self):
        h1 = LocalSpace(1, dimension=10)
        P = OperatorSymbol("P", 2)
        lhs = OperatorTrace.create(P, over_space=h1)
        rhs = 10 * P
        assert lhs == rhs


class TestOperatorMatrices(unittest.TestCase):
    def testConstruction(self):
        h1, h2, h3 = LocalSpace("h1"), LocalSpace("h2"), LocalSpace("h3")
        a, b, c = Destroy(h1), Destroy(h2), Destroy(h3)
        assert np_conjugate(a) == a.dag()

        M = Matrix([[a,b],[c,a]])
#        self.assertEqual(M.matrix, np_array([[a,b],[b,a]]))
        assert M == Matrix(np_array([[a,b],[c,a]]))
        assert M.T == Matrix(np_array([[a,c],[b,a]]))
        assert M.conjugate() == Matrix(np_array([[a.dag(),b.dag()],[c.dag(),a.dag()]]))
        assert M.H == Matrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]]))
        assert M.H == Matrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]]))

    def testMathOperations(self):
        M = Matrix([[Create("1"), 0],[0, Destroy("1")]])
        N = Matrix([[Destroy("1"), Create("2")],[0, Destroy("1")]])
        assert M+N == Matrix([[Create("1")+Destroy("1"), Create("2")],[0, 2*Destroy("1")]])
        assert M*N == Matrix([[Create("1")*Destroy("1"), Create("1")*Create("2")],[0, Destroy("1")*Destroy("1")]])
        assert IdentityOperator * M == M
        assert 1 * M == M
        assert Create("1") * identity_matrix(2) == Matrix([[Create("1"),0],[0,Create("1")]])

    def testElementExpand(self):
        assert Matrix([[(Create(1) + Create(2))*Create(3)]]).expand() == Matrix([[Create(1)*Create(3) + Create(2)*Create(3)]])


def test_op_expr_str():
    A = OperatorSymbol('A', 1)
    B = OperatorSymbol('B', 1)
    C = OperatorSymbol('C', 1)
    D = OperatorSymbol('D', 1)

    a = OperatorSymbol('a', 2)
    b = OperatorSymbol('b', 2)

    gamma = symbols('\gamma', positive=True)
    x, y = symbols('x y')

    expr = A - (gamma/2) * B + 2*C - D
    assert str(expr) == 'A + 2 * C - \\gamma/2 * B - D'

    expr = A * B + 2*(C + D)
    assert str(expr) == '2 * (C + D) + A * B'

    expr =  A * A * a * b * B
    assert str(expr) == 'A * A * B ⊗ a * b'

    expr =  (A + B) * (a + b)
    assert str(expr) == '(A + B) ⊗ (a + b)'

    expr =  (A + B) * (C + D)
    assert str(expr) == '(A + B) * (C + D)'

    expr = ((x**2 + y**2) / sqrt(2)) * (A * B + C + b)
    assert str(expr) == 'sqrt(2)*(x**2 + y**2)/2 * (C + b + A * B)'
