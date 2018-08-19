import unittest
import pytest

from numpy import (
    array as np_array, int_ as np_int, float_ as np_float)
from sympy import symbols, sqrt, I, exp, sympify

from qnet import (
    OperatorSymbol, II, IdentityOperator, ZeroOperator, OperatorPlus,
    LocalSigma, LocalProjector, OperatorTrace, Adjoint, PauliX, PauliY, PauliZ,
    ScalarTimesOperator, OperatorTimes, OperatorDerivative, Jz, Jplus, Jminus,
    Destroy, Create, Phase, Displace, Matrix, identity_matrix, LocalSpace,
    TrivialSpace, ProductSpace, FockIndex, SpinSpace, ascii, BasisNotSetError,
    adjoint, NoConjugateMatrix, BasisKet, ZeroKet, IdxSym)


def test_identity_singleton():
    """Test the Singleton properties of the IdentityOperator"""
    assert IdentityOperator() is IdentityOperator
    assert IdentityOperator.__class__() is IdentityOperator
    assert IdentityOperator.__class__.create() is IdentityOperator
    assert IdentityOperator.create() is IdentityOperator
    assert IdentityOperator.create(
        *IdentityOperator.args, **IdentityOperator.kwargs) is IdentityOperator


def test_identity_comparisons():
    """IdentityOperator only comparse to itself"""
    assert IdentityOperator is IdentityOperator
    assert IdentityOperator == IdentityOperator
    assert IdentityOperator != 1
    assert IdentityOperator != np_float(1.0)
    assert IdentityOperator != sympify(1)

    assert IdentityOperator != np_int(-3)
    assert IdentityOperator != 0.0
    assert IdentityOperator != sympify(3.5)


def test_zero_comparisons():
    """ZeroOperator only comparse to itself"""
    assert ZeroOperator is ZeroOperator
    assert ZeroOperator == ZeroOperator
    assert ZeroOperator.is_zero
    assert ZeroOperator != np_int(0)
    assert ZeroOperator != 0.0
    assert ZeroOperator != sympify(0.0)

    assert ZeroOperator != -3
    assert ZeroOperator != np_float(1.0)
    assert ZeroOperator != sympify(2)


def test_local_sigma_raise_jk():
    sig = LocalProjector(1, hs='1')
    assert sig.raise_jk() == sig
    assert sig.raise_jk(j_incr=1) == LocalSigma(2, 1, hs='1')
    assert sig.raise_jk(j_incr=-1) == LocalSigma(0, 1, hs='1')
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalSigma(1, 2, hs='1')
    assert sig.raise_jk(k_incr=-1) == LocalSigma(1, 0, hs='1')
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == LocalProjector(2, hs='1')
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == LocalProjector(0, hs='1')
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == LocalSigma(0, 2, hs='1')
    sig = LocalProjector(0, hs='1')
    assert sig.raise_jk(j_incr=1) == LocalSigma(1, 0, hs='1')
    assert sig.raise_jk(j_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalSigma(0, 1, hs='1')
    assert sig.raise_jk(k_incr=-1) == ZeroOperator
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == LocalProjector(1, hs='1')
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == ZeroOperator
    sig = LocalSigma(1, 0, hs='1')
    assert sig.raise_jk() == sig
    assert sig.raise_jk(j_incr=1) == LocalSigma(2, 0, hs='1')
    assert sig.raise_jk(j_incr=-1) == LocalProjector(0, hs='1')
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalProjector(1, hs='1')
    assert sig.raise_jk(k_incr=-1) == ZeroOperator
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == LocalSigma(2, 1, hs='1')
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == LocalSigma(0, 1, hs='1')
    # in TLS, we cannot go out of the subspace
    hil_tls = LocalSpace(label='tls', dimension=2)
    sig = LocalProjector(1, hs=hil_tls)
    assert sig.raise_jk() == sig
    assert sig.raise_jk(j_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1) == LocalSigma(0, 1, hs=hil_tls)
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == ZeroOperator
    assert sig.raise_jk(k_incr=-1) == LocalSigma(1, 0, hs=hil_tls)
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == LocalProjector(0, hs=hil_tls)
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == ZeroOperator
    sig = LocalProjector(0, hs=hil_tls)
    assert sig.raise_jk(j_incr=1) == LocalSigma(1, 0, hs=hil_tls)
    assert sig.raise_jk(j_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalSigma(0, 1, hs=hil_tls)
    assert sig.raise_jk(k_incr=-1) == ZeroOperator
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == LocalProjector(1, hs=hil_tls)
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == ZeroOperator
    sig = LocalSigma(1, 0, hs=hil_tls)
    assert sig.raise_jk() == sig
    assert sig.raise_jk(j_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1) == LocalProjector(0, hs=hil_tls)
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalProjector(1, hs=hil_tls)
    assert sig.raise_jk(k_incr=-1) == ZeroOperator
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == LocalSigma(0, 1, hs=hil_tls)
    # same for string-based labels
    hil_ge = LocalSpace(label='tls', dimension=2, basis=('g', 'e'))
    sig = LocalProjector(1, hs=hil_ge)
    assert sig == LocalProjector('e', hs=hil_ge)
    assert sig.raise_jk() == sig
    assert sig.raise_jk(j_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1) == LocalSigma(0, 1, hs=hil_ge)
    assert sig.raise_jk(j_incr=-1) == LocalSigma('g', 'e', hs=hil_ge)
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == ZeroOperator
    assert sig.raise_jk(k_incr=-1) == LocalSigma(1, 0, hs=hil_ge)
    assert sig.raise_jk(k_incr=-1) == LocalSigma('e', 'g', hs=hil_ge)
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == LocalProjector(0, hs=hil_ge)
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == ZeroOperator
    sig = LocalProjector('g', hs=hil_ge)
    assert sig.raise_jk(j_incr=1) == LocalSigma(1, 0, hs=hil_ge)
    assert sig.raise_jk(j_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalSigma(0, 1, hs=hil_ge)
    assert sig.raise_jk(k_incr=-1) == ZeroOperator
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == LocalProjector(1, hs=hil_ge)
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == ZeroOperator
    sig = LocalSigma(1, 0, hs=hil_ge)
    assert sig.raise_jk() == sig
    assert sig.raise_jk(j_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1) == LocalProjector('g', hs=hil_ge)
    assert sig.raise_jk(j_incr=-2) == ZeroOperator
    assert sig.raise_jk(k_incr=1) == LocalProjector('e', hs=hil_ge)
    assert sig.raise_jk(k_incr=-1) == ZeroOperator
    assert sig.raise_jk(k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=1, k_incr=1) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=-1) == ZeroOperator
    assert sig.raise_jk(j_incr=-2, k_incr=-2) == ZeroOperator
    assert sig.raise_jk(j_incr=-1, k_incr=1) == LocalSigma(0, 1, hs=hil_ge)
    # symbolic labels
    i = IdxSym('i')
    sig = LocalProjector(FockIndex(i), hs='1')
    assert sig.raise_jk() == sig
    assert (
        sig.raise_jk(j_incr=1) ==
        LocalSigma(FockIndex(i+1), FockIndex(i), hs='1'))
    assert (
        sig.raise_jk(j_incr=-1) ==
        LocalSigma(FockIndex(i-1), FockIndex(i), hs='1'))
    assert (
        sig.raise_jk(j_incr=-2) ==
        LocalSigma(FockIndex(i-2), FockIndex(i), hs='1'))
    assert (
        sig.raise_jk(k_incr=1) ==
        LocalSigma(FockIndex(i), FockIndex(i+1), hs='1'))
    assert (
        sig.raise_jk(k_incr=-1) ==
        LocalSigma(FockIndex(i), FockIndex(i-1), hs='1'))
    assert (
        sig.raise_jk(k_incr=-2) ==
        LocalSigma(FockIndex(i), FockIndex(i-2), hs='1'))
    assert (
        sig.raise_jk(j_incr=1, k_incr=1) ==
        LocalProjector(FockIndex(i+1), hs='1'))
    assert (
        sig.raise_jk(j_incr=-1, k_incr=-1) ==
        LocalProjector(FockIndex(i-1), hs='1'))
    assert (
        sig.raise_jk(j_incr=-1, k_incr=1) ==
        LocalSigma(FockIndex(i-1), FockIndex(i+1), hs='1'))


def test_local_sigma_reject_invalid_labels():
    """Test that a ValueError is raised when trying to instantiate a LocalSigma
    with invalid labels"""
    hs = LocalSpace('tls', basis=('g', 'e'))
    with pytest.raises(ValueError):
        LocalSigma('0', '1', hs=hs)
    with pytest.raises(ValueError):
        LocalSigma(0, 2, hs=hs)
    with pytest.raises(ValueError):
        LocalSigma(2, 0, hs=hs)
    with pytest.raises(ValueError):
        LocalSigma(0, -1, hs=hs)
    with pytest.raises(ValueError):
        LocalSigma(-1, 0, hs=hs)
    with pytest.raises(BasisNotSetError):
        LocalSigma('0', '0', hs=1)


def test_local_sigma_equivalences():
    """Test the equivalence of instantiation :class:`LocalSigma` with integer
    or str labels, if there is a defined basis"""
    hs = LocalSpace('tls', basis=('g', 'e'))
    assert LocalSigma('g', 'e', hs=hs) == LocalSigma(0, 1, hs=hs)
    assert LocalSigma(0, 'e', hs=hs) == LocalSigma(0, 1, hs=hs)
    assert LocalSigma('g', 1, hs=hs) == LocalSigma(0, 1, hs=hs)


def test_proj_create_destroy_product():
    """Test some products for Creation operator, Annihilation Operators, and
    projectors, as they typically occur during adiabatic eliminiation"""
    a = Destroy(hs="1")
    a_dag = a.dag()
    P1 = LocalProjector(1, hs=LocalSpace("1"))

    rhs = IdentityOperator + OperatorTimes(a_dag, a)
    lhs = a * a_dag
    assert lhs == rhs
    lhs = OperatorTimes.create(a, a_dag)
    assert lhs == rhs

    rhs = OperatorTimes(a_dag, a)
    lhs = a_dag * a
    assert lhs == rhs
    lhs = OperatorTimes.create(a_dag, a)
    assert lhs == rhs

    rhs = LocalSigma(1, 0, hs=LocalSpace("1"))
    lhs = P1 * a_dag
    assert lhs == rhs
    lhs = OperatorTimes.create(P1, a_dag)
    assert lhs == rhs

    rhs = P1
    lhs = P1 * a_dag * a
    assert lhs == rhs
    lhs = OperatorTimes.create(P1, a_dag, a)
    assert lhs == rhs

    rhs = 2 * P1
    assert rhs == (P1 * (1 + P1)).expand()
    lhs = P1 * a * a_dag
    assert lhs == rhs
    lhs = OperatorTimes.create(P1, a, a_dag).expand()
    assert lhs == rhs

    kappa = symbols('kappa')
    Y = (-kappa / 2) * a.dag() * a
    Id5 = OperatorPlus.create(*[LocalProjector(i, hs="1") for i in range(5)])
    prod = Id5 * Y
    res = prod.expand()
    res_expect = OperatorPlus.create(
        *[(-(i * kappa)/2) * LocalProjector(i, hs="1") for i in range(5)])
    assert res == res_expect


class TestOperatorCreation(unittest.TestCase):

    def testIdentity(self):
        assert Create(hs="1") == Create(hs="1")
        assert (OperatorSymbol.create("a", hs=1) ==
                OperatorSymbol.create("a", hs=1))

    def testImplicitHilbertSpaceCreation(self):
        hs = LocalSpace("hs")
        h2 = LocalSpace("h2")
        aa = OperatorSymbol.create("aa", hs=hs)
        bb = OperatorSymbol.create("aa", hs=hs*h2)
        a = Destroy(hs=hs)
        assert aa == OperatorSymbol.create("aa", hs="hs")
        assert bb == OperatorSymbol.create("aa", hs=("hs", "h2"))
        assert a == Destroy(hs="hs")
        assert Destroy(hs=1) == Destroy(hs="1")
        assert (OperatorSymbol.create("a", hs=1) ==
                OperatorSymbol.create("a", hs="1"))


class TestOperatorAddition(unittest.TestCase):

    def testAdditionToScalar(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol.create("a", hs=hs)
        id_ = IdentityOperator
        assert a+0 == a
        assert 0+a == a
        assert 1 + a + 1 == a + 2
        lhs = a + 2
        rhs = OperatorPlus.create(ScalarTimesOperator.create(2, id_), a)
        assert lhs == rhs

    def testOperatorOrdering(self):
        hs = LocalSpace("1")
        a = OperatorSymbol.create("a", hs=hs)
        b = OperatorSymbol.create("b", hs=hs)
        c = OperatorSymbol.create("c", hs=hs)
        assert c+b+a == OperatorPlus.create(a, b, c)

    def testAdditionToOperator(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol.create("a", hs=hs)
        b = OperatorSymbol.create("b", hs=hs)
        assert a + b == b + a
        assert a + b == OperatorPlus.create(a, b)

    def testAdditionToOperatorProduct(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol.create("a", hs=hs)
        b = OperatorSymbol.create("b", hs=hs)
        assert a + b*b*a == b*b*a + a
        assert a + b*b*a == OperatorPlus.create(a,
                                                OperatorTimes.create(b, b, a))

    def testSubtraction(self):
        hs = LocalSpace("hs")
        a = OperatorSymbol.create("a", hs=hs)
        b = OperatorSymbol.create("b", hs=hs)
        z = ZeroOperator
        assert a-a == z
        assert a-b == OperatorPlus.create(a, ScalarTimesOperator.create(-1, b))

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol.create("a", hs=h1)
        b = OperatorSymbol.create("b", hs=h2)
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
        a = OperatorSymbol.create("a", hs=h1)
        b = OperatorSymbol.create("b", hs=h2)
        c = OperatorSymbol.create("c", hs=h2)
        assert a * b == OperatorTimes.create(a,b)

        assert b * a == a * b
        assert c * a * b * c * a == OperatorTimes.create(a, a, c, b, c)


    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol.create("a", hs=h1)
        b = OperatorSymbol.create("b", hs=h2)
        assert a.space == h1
        assert (a * b).space == h1*h2


    def testEquality(self):
        assert Create(hs=1) * Create(hs=2) == Create(hs=1) * Create(hs=2)


class TestScalarTimesOperator(unittest.TestCase):
    def testZeroOne(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol.create("a", hs=h1)
        b = OperatorSymbol.create("b", hs=h2)
        z = ZeroOperator

        assert a+a == 2*a
        assert a*1 == a
        assert 1*a == a
        assert a*5 == ScalarTimesOperator.create(5, a)
        assert 5*a == a*5
        assert 2*a*3 == 6*a
        assert a*5*b == ScalarTimesOperator.create(5, a*b)
        assert a*(5*b) == ScalarTimesOperator.create(5, a*b)

        assert 0 * a == z
        assert a*0 == z
        assert 10 * z == z

    def testScalarTimesIdentity(self):
        id_ = IdentityOperator

        assert 2 * id_ == 2
        assert id_ * 2 == np_float(2.0)
        assert -3.7 * id_ == sympify(-3.7)


    def testHashability(self):
        assert hash(ScalarTimesOperator.create(1, Create(hs=1))) == \
               hash(ScalarTimesOperator.create(1, Create(hs=1)))

    def testEquality(self):
        assert 5 * Create(hs=1) == (6-1) * Create(hs=1)

    def testScalarCombination(self):
        a = OperatorSymbol.create("a", hs="h1")
        assert a+a == 2*a
        assert 3*a + 4*a == 7 * a
        assert Create(hs="1") + Create(hs="1") == 2 * Create(hs="1")

    def testHilbertSpace(self):
        h1 = LocalSpace("h1")
        h2 = LocalSpace("h2")
        a = OperatorSymbol.create("a", hs=h1)
        b = OperatorSymbol.create("b", hs=h2)
        assert (5*(a * b)).space == h1*h2

    def test_series_expand(self):
        k, l = symbols("k l")
        a = Destroy(hs="1")

        # Test for a finite polynomial with some vanishing terms
        X = (1 + l + 2 * k + k ** 3 / 2) * a
        half = sympify(1) / 2
        assert X.series_expand(k, 0, 0) == ((1 + l) * a, )
        assert X.series_expand(k, 0, 1) == ((1 + l) * a,
                                            2 * a)
        assert X.series_expand(k, 0, 2) == ((1 + l) * a,
                                            2 * a,
                                            ZeroOperator)
        assert X.series_expand(k, 0, 3) == ((1 + l) * a,
                                            2 * a,
                                            ZeroOperator,
                                            half * a)
        assert X.series_expand(k, 0, 4) == ((1 + l) * a,
                                            2 * a,
                                            ZeroOperator,
                                            half * a,
                                            ZeroOperator)

        # Test for an infinite series, with expansion around nonzero center
        Y = -a / k
        for n in range(6):
            assert Y.series_expand(k, -1.0, n) == (a, ) * (n + 1)

        # Test at a singularity
        with pytest.raises(ValueError):
            Y.series_expand(k, 0, 0)


class TestDifferentiation(unittest.TestCase):

    def testConstantOps(self):
        x = symbols("x")

        X = OperatorSymbol.create("X", hs=1)
        assert X.diff(x) == ZeroOperator
        assert (2*X).diff(x) == ZeroOperator
        assert X.dag().diff(x) == ZeroOperator

        a = Destroy(hs=1)
        assert a.diff(x) == ZeroOperator
        assert a.dag().diff(x) == ZeroOperator
        assert (a + a.dag()).diff(x) == ZeroOperator
        assert (a * a.dag()).diff(x) == ZeroOperator

        s = LocalSigma.create(1, 2, hs=1)
        assert s.diff(x) == ZeroOperator

        alpha = symbols('alpha')
        Ph = Phase.create(alpha, hs=1)
        assert Ph.diff(x) == ZeroOperator
        assert isinstance(Ph.diff(alpha), OperatorDerivative)

    def testNonConstantOps(self):
        x = symbols("x", real=True)

        X = OperatorSymbol.create("X", hs=1)
        Y = OperatorSymbol.create("Y", hs=1)
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
        h = SpinSpace('h', spin=j)
        jz = Jz(hs=h)
        jp = Jplus(hs=h)
        jm = Jminus(hs=h)

        assert (jp*jm-jm*jp).expand() == 2*jz
        assert (jz*jm-jm*jz).expand() == -jm
        assert (jz*jp-jp*jz).expand() == jp

        assert jp*LocalProjector('+3', hs=h) == ZeroOperator
        assert (jp*LocalProjector('+2', hs=h) ==
                sqrt(j*(j+1)-2*(2+1)) * LocalSigma.create('+3', '+2', hs=h))

        assert jm*LocalProjector('-3', hs=h) == ZeroOperator
        assert (jm*LocalProjector('-2', hs=h) ==
                sqrt(j*(j+1)-2*(2+1)) * LocalSigma.create('-3', '-2', hs=h))

        assert jz*LocalProjector('-3', hs=h) == -3*LocalProjector('-3', hs=h)

        assert LocalProjector('+3', hs=h)*jm == ZeroOperator
        assert (LocalProjector('+2', hs=h)*jm ==
                sqrt(j*(j+1)-2*(2+1))*LocalSigma.create('+2', '+3', hs=h))

        assert LocalProjector('-3', hs=h)*jp == ZeroOperator
        assert (LocalProjector('-2', hs=h)*jp ==
                sqrt(j*(j+1)-2*(2+1))*LocalSigma.create('-2', '-3', hs=h))

        assert LocalProjector('-3', hs=h)*jz == -3*LocalProjector('-3', hs=h)

        tls = SpinSpace('tls', spin='1/2', basis=('-', '+'))
        sz = Jz(hs=tls)
        sp = Jplus(hs=tls)
        sm = Jminus(hs=tls)

        assert (sp*sm-sm*sp).expand() == 2*sz
        assert (sz*sm-sm*sz).expand() == -sm
        assert (sz*sp-sp*sz).expand() == sp

    def testPhase(self):
        assert Phase.create(5, hs=1).adjoint() == Phase.create(-5, hs=1)
        assert (Phase.create(5, hs=1) * Phase.create(-5, hs=1) ==
                IdentityOperator)
        assert (Phase.create(5, hs=1) * Create(hs=1) *
                Phase.create(-5, hs=1) == exp(I * 5) * Create(hs=1))
        assert (Phase.create(5, hs=1) * LocalSigma.create(3, 4, hs=1) ==
                exp(15 * I) * LocalSigma.create(3,4, hs=1))
        assert (LocalSigma.create(3,4, hs=1) * Phase.create(5, hs=1) ==
                exp(20 * I) * LocalSigma.create(3,4, hs=1))
        assert (Phase.create(5, hs=1) * LocalSigma.create(0,4, hs=1) ==
                LocalSigma.create(0,4, hs=1))
        assert (LocalSigma.create(3,0, hs=1) * Phase.create(5, hs=1) ==
                LocalSigma.create(3,0, hs=1))

    def testDisplace(self):
        assert (Displace.create(5+6j, hs=1).adjoint() ==
                Displace.create(-5-6j, hs=1))
        assert (Displace.create(5+6j, hs=1) * Displace.create(-5-6j, hs=1) ==
                IdentityOperator)
        assert (Displace.create(5+6j, hs=1) * Create(hs=1) *
                Displace.create(-5-6j, hs=1) == Create(hs=1) - (5-6j))

    def testLocalSigmaPi(self):
        h = LocalSpace("h")
        assert (
            LocalSigma.create(0, 1, hs=h) * LocalSigma.create(1, 2, hs=h) ==
            LocalSigma.create(0, 2, hs=h))
        assert LocalSigma.create(0, 0, hs=h) == LocalProjector(0, hs=h)

    def testAnnihilation(self):
        h = LocalSpace("h")
        z = ZeroOperator
        assert Destroy(hs=h) * LocalSigma.create(0, 1, hs=h) == z
        assert LocalSigma.create(1, 0, hs=h) * Create(hs=h) == z


class TestOperatorTrace(unittest.TestCase):

    def testConstruction(self):
        M = OperatorSymbol.create("M", hs=LocalSpace(1))
        N = OperatorSymbol.create("N", hs=ProductSpace(LocalSpace(1),
                                                       LocalSpace(2)))
        assert (
            OperatorTrace.create(M, over_space=LocalSpace(1)) ==
            OperatorTrace(M, over_space=LocalSpace(1)))
        assert (
            OperatorTrace.create(M, over_space=LocalSpace(1)).space ==
            TrivialSpace)
        assert (
            OperatorTrace.create(N, over_space=LocalSpace(1)).space ==
            LocalSpace(2))

    def testSimplificationPlus(self):
        M = OperatorSymbol.create("M", hs=1)
        N = OperatorSymbol.create("N", hs=1)
        O = OperatorSymbol.create("O", hs=1)

        assert (OperatorTrace.create(M+N, over_space=1) ==
                (OperatorTrace.create(M, over_space=1) +
                 OperatorTrace.create(N, over_space=1)))
        assert (OperatorTrace.create((M+N)*O, over_space=1).expand() ==
                (OperatorTrace.create(M*O, over_space=1) +
                 OperatorTrace.create(N*O, over_space=1)))

    def testSimplificationTimes(self):
        hs1 = LocalSpace(1)
        hs2 = LocalSpace(2)
        hs3 = LocalSpace(3)
        hs_123 = hs1 * hs2 * hs3
        M = OperatorSymbol.create("M", hs=hs1)
        N = OperatorSymbol.create("N", hs=hs2)
        O = OperatorSymbol.create("O", hs=hs_123)
        assert (OperatorTrace.create(M * N, over_space=hs1) ==
                OperatorTrace.create(M, over_space=hs1) * N)
        lhs = OperatorTrace.create(
                     M*N*O, over_space=ProductSpace(hs2, hs3))
        rhs = M * OperatorTrace.create(
            N * OperatorTrace.create(O, over_space=hs3), over_space=hs2)
        assert lhs == rhs
        assert (OperatorTrace.create(
                    OperatorTrace.create(N, over_space=hs2) * M,
                    over_space=hs1
                ) == (
                    OperatorTrace.create(M, over_space=hs1) *
                    OperatorTrace.create(N, over_space=hs2)
                ))
        assert (OperatorTrace.create(
                    M * N,
                    over_space=ProductSpace(hs1, hs2)
                ) == (
                    (OperatorTrace.create(M, over_space=hs1) *
                     OperatorTrace.create(N, over_space=hs2))
                ))

    def testSimplificationScalarTimesOperator(self):
        M = OperatorSymbol.create("M", hs=LocalSpace(1))
        assert (OperatorTrace.create(10 * M, over_space=1) ==
                10 * OperatorTrace.create(M, over_space=1))

    def testSimplificationAdjoint(self):
        M = OperatorSymbol.create("M", hs=1)
        lhs = OperatorTrace.create(M.adjoint(), over_space=1)
        rhs = Adjoint(OperatorTrace.create(M, over_space=1))
        assert lhs == rhs

    def testLocalOps(self):
        op = OperatorTrace.create(Create(hs=1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(Destroy(hs=1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma.create(1, 2, hs=1), over_space=1)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma.create(1, 1, hs=1), over_space=1)
        assert op == IdentityOperator
        hs = LocalSpace(1, basis=('g', 'e'))
        op = OperatorTrace.create(LocalSigma.create('e', 'g', hs=hs), over_space=hs)
        assert op == ZeroOperator
        op = OperatorTrace.create(LocalSigma.create('e', 'e', hs=hs), over_space=hs)
        assert op == IdentityOperator

    def testSimplificationMaxwellBloch(self):

        a = LocalSpace("a", basis=('h','g'))
        f = LocalSpace("f")
        x,y,z = symbols("x,y,z", real = True)
        alpha = symbols("alpha")
        rho_a = (II + x * PauliX(a) + y * PauliY(a) + z * PauliZ(a)) / 2
        sigma = PauliX(a) + I*PauliY(a)
        rho_f = (Displace.create(alpha, hs=f) * LocalProjector(0, hs=f) *
                 Displace.create(-alpha, hs=f))
        rho = rho_a * rho_f
        lhs = OperatorTrace.create(rho, over_space=ProductSpace(a, f))
        lhs = lhs.expand()
        assert lhs == IdentityOperator

    def testDimensionPrefactor(self):
        h1 = LocalSpace(1, dimension=10)
        P = OperatorSymbol.create("P", hs=2)
        lhs = OperatorTrace.create(P, over_space=h1)
        rhs = 10 * P
        assert lhs == rhs


def test_opmatrix_construction():
    h1, h2, h3 = LocalSpace("h1"), LocalSpace("h2"), LocalSpace("h3")
    a, b, c = Destroy(hs=h1), Destroy(hs=h2), Destroy(hs=h3)

    M = Matrix([[a, b], [c, a]])
    assert M == Matrix(np_array([[a, b], [c, a]]))
    assert M.T == Matrix(np_array([[a, c], [b, a]]))
    with pytest.raises(NoConjugateMatrix):
        M.conjugate()
    assert (
        M.element_wise(adjoint) ==
        Matrix(np_array([[a.dag(), b.dag()], [c.dag(), a.dag()]])))
    assert M.H == Matrix(np_array([[a.dag(), c.dag()], [b.dag(), a.dag()]]))
    assert M.H == M.adjoint()


def test_opmatrix_math_operations():
    M = Matrix([[Create(hs="1"), 0], [0, Destroy(hs="1")]])
    N = Matrix([[Destroy(hs="1"), Create(hs="2")], [0, Destroy(hs="1")]])

    sum = M + N
    assert isinstance(sum, Matrix)
    assert sum.shape == (2, 2)
    assert sum[0, 0] == Create(hs="1") + Destroy(hs="1")
    assert sum[0, 1] == Create(hs="2")
    assert sum[1, 0] == 0
    assert sum[1, 1] == 2 * Destroy(hs="1")

    assert (
        M * N == Matrix([
            [Create(hs="1")*Destroy(hs="1"), Create(hs="1")*Create(hs="2")],
            [ZeroOperator, Destroy(hs="1")*Destroy(hs="1")]]))
    assert (
        IdentityOperator * M == Matrix([
            [Create(hs="1"), ZeroOperator],
            [ZeroOperator, Destroy(hs="1")]]))
    assert 1 * M == M
    assert (
        Create(hs="1") * identity_matrix(2) == Matrix([
            [Create(hs="1"), ZeroOperator],
            [ZeroOperator, Create(hs="1")]]))


def test_opmatrix_element_expand():
    assert (
        Matrix([[(Create(hs=1) + Create(hs=2)) * Create(hs=3)]]).expand() ==
        Matrix([[Create(hs=1)*Create(hs=3) + Create(hs=2)*Create(hs=3)]]))


def test_local_operator_init():
    """Test that LocalOperators with different labels can be
    distinguished"""
    x = OperatorSymbol(label='x', hs=0)
    p = OperatorSymbol(label='p', hs=0)
    assert x != p


def test_custom_identifier():
    """Test that rule application preserves custom local identifiers, and that
    'Create' and 'Destroy' share the same identifier"""
    hilbert_spaces = [
        LocalSpace(0, local_identifiers={'Create': 'b', 'Destroy': 'b'}),
        LocalSpace(0, local_identifiers={'Create': 'b'}),
        LocalSpace(0, local_identifiers={'Destroy': 'b'})]
    for hs in hilbert_spaces:
        b = Destroy(hs=hs)
        expr = b * b.dag()
        assert ascii(expr) == '1 + b^(0)H * b^(0)'
        expr = expr.substitute({hs: LocalSpace(0)})
        assert ascii(expr) == '1 + a^(0)H * a^(0)'


def test_create_destroy_product_expand():
    a = Destroy(hs=1)
    a_dag = Create(hs=1)

    expr = a * a * a_dag * a_dag
    result = expr.expand()
    expected = 4 * a_dag * a + a_dag * a_dag * a * a + 2
    assert result == expected

    expr = a * a * a * a_dag * a_dag * a_dag
    result = expr.expand()
    expected = (
        6 +
        18 * a_dag * a +
        9 * a_dag * a_dag * a * a +
        a_dag * a_dag * a_dag * a * a * a)
    assert result == expected


def test_issue76():
    """Test resolution of #76"""
    ket_0 = BasisKet(0, hs=0)
    ket_1 = BasisKet(1, hs=0)
    bra_0 = ket_0.dag()
    bra_1 = ket_1.dag()
    sig_10 = LocalSigma(1, 0, hs=0)
    sig_01 = LocalSigma(0, 1, hs=0)
    sig_11 = LocalSigma(1, 1, hs=0)
    P0 = LocalProjector(0, hs=0)
    P1 = LocalProjector(1, hs=0)
    ZeroBra = ZeroKet.dag()

    # first, verify LocalSigma

    # ket
    expr = sig_10 * ket_1
    assert expr == ZeroKet
    expr = sig_01 * ket_1
    assert expr == ket_0
    expr = sig_11 * ket_1
    assert expr == ket_1
    # bra
    expr = bra_1 * sig_10
    assert expr == bra_0
    expr = bra_1 * sig_01
    assert expr == ZeroBra
    expr = bra_1 * sig_11
    assert expr == bra_1

    # second, LocalProjector
    # (Mote: at the time of the issue, LocalProjector was a subclass of
    # LocalSigma. The resolution was to make it a function only)

    # ket
    expr = P1 * ket_1
    assert expr == ket_1
    expr = P0 * ket_1
    assert expr == ZeroKet
    # bra
    expr = bra_1 * P1
    assert expr == bra_1
    expr = bra_1 * P0
    assert expr == ZeroBra
