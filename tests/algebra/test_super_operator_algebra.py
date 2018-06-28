import unittest
from sympy import symbols, sqrt, I, conjugate


from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.super_operator_algebra import (
        SuperOperator, SuperOperatorSymbol, IdentitySuperOperator,
        SuperOperatorPlus, SuperOperatorTimes, SuperOperatorTimesOperator,
        ScalarTimesSuperOperator, ZeroSuperOperator, SPre, SPost, liouvillian,
        liouvillian_normal_form)
from qnet.algebra.pattern_matching import wc, ProtoExpr, pattern_head
from qnet.algebra.core.operator_algebra import (
        OperatorPlus, ScalarTimesOperator, OperatorSymbol, ZeroOperator,
    OperatorTimes)
from qnet.algebra.library.fock_operators import Destroy, Create


class TestSuperOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        assert SuperOperatorSymbol("a", hs=1) == SuperOperatorSymbol("a", hs=1)

    def testMatch(self):

        A = wc("A", head=SuperOperator)
        a = SuperOperatorSymbol("a", hs="hs")
        b = SuperOperatorSymbol("b", hs="hs")
        b2 = SuperOperatorSymbol("b", hs="hs")

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
        a = SuperOperatorSymbol("a", hs=hs)
        id = IdentitySuperOperator
        assert a+0 == a
        assert 0+a == a
        assert 1 + a + 1 == a + 2
        assert a + 2 == SuperOperatorPlus(ScalarTimesSuperOperator(2,id),a)


    def testAdditionToSuperOperator(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        assert a + b == b + a
        assert a + b == SuperOperatorPlus(a,b)
        assert (a+b).space == h1 * h2

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = SuperOperatorSymbol("a", hs=hs)
        b = SuperOperatorSymbol("b", hs=hs)
        z = ZeroSuperOperator
        assert a-a == z
        assert a-b == SuperOperatorPlus(a, ScalarTimesSuperOperator(-1,b))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        assert (a+b).space == h1 * h2


    def testCommutativity(self):
        h1 = LocalSpace("h1")
        assert (SuperOperatorSymbol("A", hs=h1) +
                SuperOperatorSymbol("B", hs=h1) ==
               (SuperOperatorSymbol("B", hs=h1) +
                SuperOperatorSymbol("A", hs=h1)))




class TestSuperOperatorTimes(unittest.TestCase):

    def testIdentity(self):
        h1 = LocalSpace("h1")
        a = SuperOperatorSymbol("a", hs=h1)
        id = IdentitySuperOperator
        assert a * id == a
        assert id * a == a

    def testOrdering(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        c = SuperOperatorSymbol("c", hs=h2)
        dpre = SPre(SuperOperatorSymbol("d", hs=h1))
        epre = SPre(SuperOperatorSymbol("e", hs=h1))
        dpost = SPost(SuperOperatorSymbol("d", hs=h1))
        epost = SPost(SuperOperatorSymbol("e", hs=h1))

        assert a * b == SuperOperatorTimes(a,b)
        assert b * a == a * b
        assert c * a * b * c * a == SuperOperatorTimes(a, a, c, b, c)

    def testSPreSPostRules(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        d = OperatorSymbol("d", hs=h1)
        e = OperatorSymbol("e", hs=h1)
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
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        assert a.space == h1
        assert (a * b).space == h1*h2


    def testCommutativity(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        assert a*b == b*a




class TestScalarTimesSuperOperator(unittest.TestCase):

    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
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
        a = SuperOperatorSymbol("a", hs=h1)
        assert 5*a == ScalarTimesSuperOperator(5, a)

    def testScalarCombination(self):
        a = SuperOperatorSymbol("a", hs="h1")
        assert a+a == 2*a
        assert 3*a + 4*a == 7 * a

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        assert (5*(a * b)).space == h1*h2


class TestSuperOperatorTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol("a", hs=h1)
        B = SuperOperatorSymbol("B", hs=h2)
        z = ZeroSuperOperator
        one = IdentitySuperOperator

        assert one * a == a
        assert z * a == ZeroOperator
        # self.assertEqual(B * a, a * (B * IdentityOperator))

    def testEqual2(self):
        h1 = LocalSpace("h1")
        A = SuperOperatorSymbol("A", hs=h1)
        a = OperatorSymbol("a", hs=h1)

        OTO = SuperOperatorTimesOperator(A, a)
        assert A * a == OTO

    def testCombination(self):

        h1 = LocalSpace("h1")
        a = OperatorSymbol("a", hs=h1)
        A = SuperOperatorSymbol("A", hs=h1)
        B = SuperOperatorSymbol("B", hs=h1)
        assert A * (B * a) == (A * B) * a



    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = SuperOperatorSymbol("a", hs=h1)
        b = SuperOperatorSymbol("b", hs=h2)
        assert (5 * (a * b)).space == h1 * h2


def test_liouvillian_normal_form():
    kappa_1, kappa_2 = symbols('kappa_1, kappa_2', positive=True)
    Delta = symbols('Delta', real=True)
    alpha = symbols('alpha')
    H = (Delta * Create(hs=1) * Destroy(hs=1) +
         (sqrt(kappa_1) / (2 * I)) *
         (alpha * Create(hs=1) - alpha.conjugate() * Destroy(hs=1)))
    Ls = [sqrt(kappa_1) * Destroy(hs=1) + alpha, sqrt(kappa_2) * Destroy(hs=1)]
    LL = liouvillian(H, Ls)
    Hnf, Lsnf = liouvillian_normal_form(LL)
    Hnf_expected = OperatorPlus(
            ScalarTimesOperator(-I*alpha*sqrt(kappa_1),
                                Create(hs=1)),
            ScalarTimesOperator(I*sqrt(kappa_1)*conjugate(alpha),
                                Destroy(hs=1)),
            ScalarTimesOperator(Delta,
                                OperatorTimes(Create(hs=1), Destroy(hs=1))))
    assert Hnf == Hnf_expected
    Lsnf_expected = [ScalarTimesOperator(sqrt(kappa_1 + kappa_2),
                                         Destroy(hs=1))]
    assert Lsnf == Lsnf_expected
