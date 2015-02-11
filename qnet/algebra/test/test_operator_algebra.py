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
from qnet.algebra.operator_algebra import *


class TestOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        self.assertEqual(Create("1"), Create("1"))
        self.assertEqual(OperatorSymbol("a", 1), OperatorSymbol("a",1))

    def testImplicitHilbertSpaceCreation(self):
        hs = local_space("hs")
        h2 = local_space("h2")
        aa = OperatorSymbol("aa", hs)
        bb = OperatorSymbol("aa", hs * h2)
        a = Destroy(hs)
        self.assertEqual(aa, OperatorSymbol("aa", "hs"))
        self.assertEqual(bb, OperatorSymbol("aa", ("hs", "h2")))
        self.assertEqual(a, Destroy("hs"))
        self.assertEqual(Destroy(1), Destroy("1"))
        self.assertEqual(OperatorSymbol("a", 1), OperatorSymbol("a","1"))

    def testMatch(self):

        A = wc("A", head = Operator)
        a = Create("1")
        c = Create("1")
        b = OperatorSymbol("b","hs")
        b2 = OperatorSymbol("b","hs")
        n = wc("n", head = int)
        m = wc("m", head = int)

        self.assertEqual(a,c)
        self.assertEqual(b,b2)
        self.assertEqual(match(A, a),Match(A = a))
        self.assertEqual(match(A, b),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((b, b))),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((b, b2))),Match(A = b))
        self.assertEqual(match(PatternTuple((A,A)), OperandsTuple((a, c))),Match(A = a))
        self.assertEqual(match(PatternTuple((Phase(ls, u), LocalSigma(ls, n, m))), OperandsTuple((Phase.create(1, 5),LocalSigma.create(1, 3, 4)))), Match(ls = local_space(1), u = 5, n = 3, m = 4))


class TestOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        id = IdentityOperator
        self.assertEqual(a+0, a)
        self.assertEqual(0+a, a)
        self.assertEqual(1 + a + 1, a + 2)
        self.assertEqual(a + 2, OperatorPlus(ScalarTimesOperator(2, id), a))

    def testOperatorOrdering(self):
        hs = local_space("1")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        c = OperatorSymbol("c", hs)
        self.assertEqual(c+b+a, OperatorPlus(a, b, c))

    def testAdditionToOperator(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        self.assertEqual(a + b, b + a)
        self.assertEqual(a + b, OperatorPlus(a, b))
    
    def testAdditionToOperatorProduct(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        self.assertEqual(a + b*b*a, b*b*a + a)
        self.assertEqual(a + b*b*a, OperatorPlus(a, OperatorTimes(b, b, a)))

    def testSubtraction(self):
        hs = local_space("hs")
        a = OperatorSymbol("a", hs)
        b = OperatorSymbol("b", hs)
        z = ZeroOperator
        self.assertEqual(a-a, z)
        self.assertEqual(a-b, OperatorPlus(a, ScalarTimesOperator(-1, b)))

    def testHilbertSpace(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        b = OperatorSymbol("b", h2)
        self.assertEqual(a.space, h1)
        self.assertEqual((a + b).space, h1*h2)

    def testEquality(self):
        self.assertEqual(Create(1)+Create(2), Create(2)+Create(1))





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


    def testEquality(self):
        self.assertEqual(Create(1)*Create(2), Create(1)*Create(2))





class TestScalarTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = local_space("h1")
        h2 = local_space("h2")
        a = OperatorSymbol("a", h1)
        b =  OperatorSymbol("b", h2)
        z = ZeroOperator

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

    def testEquality(self):
        self.assertEqual(5*Create(1), (6-1)* Create(1))

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
    def testCommutatorAAdag(self):
        h = local_space("h")
        ii = IdentityOperator
        self.assertEqual(Destroy(h) * Create(h) - Create(h) * Destroy(h), ii )

    def testPhase(self):
        self.assertEqual(Phase(1, 5).adjoint(), Phase(1, -5))
        self.assertEqual(Phase(1, 5) * Phase(1, -5), IdentityOperator)
        self.assertEqual(Phase(1, 5) * Create(1) * Phase(1, -5), exp(I * 5) * Create(1))
        self.assertEqual(Phase(1, 5) * LocalSigma(1, 3, 4), exp(15 * I) * LocalSigma(1,3,4))
        self.assertEqual(LocalSigma(1,3,4) * Phase(1, 5), exp(20 * I) * LocalSigma(1,3,4))
        self.assertEqual(Phase(1, 5) * LocalSigma(1,0,4), LocalSigma(1,0,4))
        self.assertEqual(LocalSigma(1,3,0) * Phase(1, 5), LocalSigma(1,3,0))

    def testDisplace(self):
        self.assertEqual(Displace(1, 5 + 6j).adjoint(), Displace(1, -5-6j))
        self.assertEqual(Displace(1, 5+6j) * Displace(1, -5-6j), IdentityOperator)
        self.assertEqual(Displace(1, 5+6j) * Create(1) * Displace(1, -5-6j), Create(1) - (5-6j))

#    def testSqueeze(self):
#        self.assertEqual(Squeeze(1, 5 + 6j).adjoint(), Squeeze(1, -5-6j))
#        self.assertEqual(Squeeze(1, 5+6j) * Squeeze(1, -5-6j), 1)
#        self.assertEqual(Squeeze(1, 1 + I) * Create(1) * Squeeze(1, -1 - I), cosh(sqrt(2)) * Create(1) - sinh(sqrt(2)) * exp(-I*pi/4)* Destroy(1))


    def testLocalSigmaPi(self):
        h = local_space("h")
        self.assertEqual(LocalSigma(h, 0, 1)* LocalSigma(h, 1, 2), LocalSigma(h, 0, 2))
        self.assertEqual(LocalSigma(h, 0, 0), LocalProjector(h, 0))

    def testAnnihilation(self):
        h = local_space("h")
        z = ZeroOperator
        self.assertEqual(Destroy(h) * LocalSigma(h, 0, 1), z)
        self.assertEqual(LocalSigma(h, 1, 0) * Create(h), z)


class TestOperatorTrace(unittest.TestCase):

    def testConstruction(self):
        M = OperatorSymbol("M", 1)
        N = OperatorSymbol("N", ProductSpace(local_space(1), local_space(2)))
        self.assertEqual(OperatorTrace.create(1, M), OperatorTrace(1, M))
        self.assertEqual(OperatorTrace(1, M).space, TrivialSpace)
        self.assertEqual(OperatorTrace(1, N).space, local_space(2))

    def testSimplificationPlus(self):
        M = OperatorSymbol("M", 1)
        N = OperatorSymbol("N", 1)
        O = OperatorSymbol("O", 1)

        self.assertEqual(OperatorTrace.create(1, M+N), OperatorTrace(1, M) + OperatorTrace(1,N))
        self.assertEqual(OperatorTrace.create(1, (M+N)*O).expand(), OperatorTrace(1, M*O) + OperatorTrace(1,N*O))

    def testSimplificationTimes(self):
        M = OperatorSymbol("M", 1)
        N = OperatorSymbol("N", 2)
        O = OperatorSymbol("O", ProductSpace(local_space(1), local_space(2), local_space(3)))
        self.assertEqual(OperatorTrace.create(1, M * N), OperatorTrace(1, M) * N)
        self.assertEqual(OperatorTrace.create(ProductSpace(local_space(2),local_space(3)), M*N*O), M * OperatorTrace(2, N * OperatorTrace(3,O)))
        self.assertEqual(match(PatternTuple((wc("h", head = LocalSpace), wc("A", head = Operator))), OperandsTuple((local_space(1),OperatorTrace.create(2, N) * M))), Match(h = local_space(1), A = (OperatorTrace.create(2, N) * M)))
        self.assertEqual(OperatorTrace.create(1, OperatorTrace.create(2, N) * M), OperatorTrace(1, M) * OperatorTrace(2, N))
        self.assertEqual(OperatorTrace.create(ProductSpace(local_space(1),local_space(2)), M * N), OperatorTrace(1, M) * OperatorTrace(2,N))

    def testSimplificationScalarTimesOperator(self):
        M = OperatorSymbol("M", 1)
        self.assertEqual(OperatorTrace.create(1, 10 * M), 10 * OperatorTrace(1, M))

    def testSimplificationAdjoint(self):
        M = OperatorSymbol("M", 1)
        self.assertEqual(OperatorTrace.create(1, M.adjoint()), Adjoint(OperatorTrace(1, M)))


    def testLocalOps(self):
        self.assertEqual(OperatorTrace.create(1, Create(1)), ZeroOperator)
        self.assertEqual(OperatorTrace.create(1, Destroy(1)), ZeroOperator)
        self.assertEqual(OperatorTrace.create(1, LocalSigma(1, 1,2)), ZeroOperator)
        self.assertEqual(OperatorTrace.create(1, LocalSigma(1, 1,1)), IdentityOperator)
        self.assertEqual(OperatorTrace.create(1, LocalSigma(1, 'e','g')), ZeroOperator)
        self.assertEqual(OperatorTrace.create(1, LocalSigma(1, 'e','e')), IdentityOperator)

    def testSimplificationMaxwellBloch(self):

        a = local_space("a", basis=('h','g'))
        f = local_space("f")
        from sympy import symbols
        x,y,z = symbols("x,y,z", real = True)
        alpha = symbols("alpha")
        rho_a = (IdentityOperator + x * X(a) + y * Y(a) + z * Z(a)) / 2
        sigma = X(a) + I*Y(a)
        rho_f = Displace(f, alpha) * LocalProjector(f, 0) * Displace(f, -alpha)
        rho = rho_a * rho_f


        self.assertEqual(OperatorTrace.create(ProductSpace(a, f), rho).expand(), IdentityOperator)



    def testDimensionPrefactor(self):
        h1 = local_space(1, dimension = 10)
        P = OperatorSymbol("P", 2)
        self.assertEqual(OperatorTrace.create(h1, P), 10 * P)

class TestOperatorMatrices(unittest.TestCase):
    def testConstruction(self):
        h1, h2, h3 = local_space("h1"), local_space("h2"), local_space("h3")
        a, b, c = Destroy(h1), Destroy(h2), Destroy(h3)
        self.assertEqual(np_conjugate(a), a.dag())

        M = Matrix([[a,b],[c,a]])
#        self.assertEqual(M.matrix, np_array([[a,b],[b,a]]))
        self.assertEqual(M, Matrix(np_array([[a,b],[c,a]])))
        self.assertEqual(M.T, Matrix(np_array([[a,c],[b,a]])))
        self.assertEqual(M.conjugate(), Matrix(np_array([[a.dag(),b.dag()],[c.dag(),a.dag()]])))
        self.assertEqual(M.H, Matrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]])))
        self.assertEqual(M.H, Matrix(np_array([[a.dag(),c.dag()],[b.dag(),a.dag()]])))

    def testMathOperations(self):
        M = Matrix([[Create("1"), 0],[0, Destroy("1")]])
        N = Matrix([[Destroy("1"), Create("2")],[0, Destroy("1")]])
        self.assertEqual(M+N, Matrix([[Create("1")+Destroy("1"), Create("2")],[0, 2*Destroy("1")]]))
        self.assertEqual(M*N, Matrix([[Create("1")*Destroy("1"), Create("1")*Create("2")],[0, Destroy("1")*Destroy("1")]]))
        self.assertEqual(IdentityOperator * M, M)
        self.assertEqual(1 * M, M)
        self.assertEqual(Create("1") * identity_matrix(2), Matrix([[Create("1"),0],[0,Create("1")]]))


    def testElementExpand(self):
        self.assertEqual(Matrix([[(Create(1) + Create(2))*Create(3)]]).expand(), Matrix([[Create(1)*Create(3) + Create(2)*Create(3)]]))




