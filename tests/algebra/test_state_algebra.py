import unittest

from sympy import sqrt, exp, I, pi, IndexedBase, symbols, factorial

from qnet.algebra.core.abstract_algebra import _apply_rules
from qnet.algebra.core.scalar_algebra import (
    ScalarValue, KroneckerDelta, Zero, One)
from qnet.algebra.toolbox.core import temporary_rules
from qnet.algebra.core.operator_algebra import (
        OperatorSymbol, LocalSigma, IdentityOperator, OperatorPlus)
from qnet.algebra.library.spin_algebra import (
    Jz, Jplus, Jminus, SpinSpace,SpinBasisKet)

from qnet.algebra.library.fock_operators import (
    Destroy, Create, Phase,
    Displace)
from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.state_algebra import (
    KetSymbol, ZeroKet, KetPlus, ScalarTimesKet, CoherentStateKet,
    TrivialKet, TensorKet, BasisKet, Bra, OperatorTimesKet, BraKet,
    KetBra, KetIndexedSum)
from qnet.algebra.core.exceptions import UnequalSpaces
from qnet.utils.indices import (
    IdxSym, FockIndex, IntIndex, StrLabel, FockLabel, SymbolicLabelBase,
    IndexOverFockSpace, IndexOverRange, SpinIndex)
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
        assert z != 0
        assert z.is_zero

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
        hs1 = LocalSpace(1)
        assert (
            Create(hs=hs1) * BasisKet(2, hs=hs1) ==
            sqrt(3) * BasisKet(3, hs=hs1))
        assert (
            Destroy(hs=hs1) * BasisKet(2, hs=hs1) ==
            sqrt(2) * BasisKet(1, hs=hs1))
        assert (
            Destroy(hs=hs1) * BasisKet(0, hs=hs1) == ZeroKet)
        coh = CoherentStateKet(10., hs=hs1)
        a = Destroy(hs=hs1)
        lhs = a * coh
        rhs = 10 * coh
        assert lhs == rhs

    def testSpin(self):
        j = 3
        h = SpinSpace('j', spin=j)
        assert (Jplus(hs=h) * BasisKet('+2', hs=h) ==
                sqrt(j*(j+1)-2*(2+1)) * BasisKet('+3', hs=h))
        assert (Jminus(hs=h) * BasisKet('+2', hs=h) ==
                sqrt(j*(j+1)-2*(2-1)) * BasisKet('+1', hs=h))
        assert Jz(hs=h) * BasisKet('+2', hs=h) == 2 * BasisKet('+2', hs=h)

        tls = SpinSpace('tls', spin='1/2', basis=('-', '+'))
        assert (
            Jplus(hs=tls) * BasisKet('-', hs=tls) == BasisKet('+', hs=tls))
        assert (
            Jminus(hs=tls) * BasisKet('+', hs=tls) == BasisKet('-', hs=tls))
        assert (
            Jz(hs=tls) * BasisKet('+', hs=tls) == BasisKet('+', hs=tls) / 2)
        assert (
            Jz(hs=tls) * BasisKet('-', hs=tls) == -BasisKet('-', hs=tls) / 2)

    def testPhase(self):
        hs1 = LocalSpace(1)
        assert (Phase(5, hs=hs1) * BasisKet(3, hs=hs1) ==
                exp(I * 15) * BasisKet(3, hs=hs1))
        lhs = Phase(pi, hs=hs1) * CoherentStateKet(3., hs=hs1)
        rhs = CoherentStateKet(-3., hs=hs1)
        assert lhs.__class__ == rhs.__class__
        assert lhs.space == rhs.space
        assert abs(lhs.ampl - rhs.ampl) < 1e-14

    def testDisplace(self):
        hs1 = LocalSpace(1)
        assert (Displace(5 + 6j, hs=hs1) * CoherentStateKet(3., hs=hs1) ==
                exp(I * ((5+6j)*3).imag) * CoherentStateKet(8 + 6j, hs=hs1))
        assert (Displace(5 + 6j, hs=hs1) * BasisKet(0, hs=hs1) ==
                CoherentStateKet(5+6j, hs=hs1))

    def testLocalSigmaPi(self):
        assert (LocalSigma(0, 1, hs=1) * BasisKet(1, hs=1) ==
                BasisKet(0, hs=1))
        assert (LocalSigma(0, 0, hs=1) * BasisKet(1, hs=1) ==
                ZeroKet)

    def testActLocally(self):
        hs1 = LocalSpace(1)
        hs2 = LocalSpace(2)
        assert ((Create(hs=hs1) * Destroy(hs=hs2)) *
                (BasisKet(2, hs=hs1) * BasisKet(1, hs=hs2)) ==
                sqrt(3) * BasisKet(3, hs=hs1) * BasisKet(0, hs=hs2))

    def testOperatorTensorProduct(self):
        hs1 = LocalSpace(1)
        hs2 = LocalSpace(2)
        assert ((Create(hs=hs1)*Destroy(hs=hs2)) *
                (BasisKet(0, hs=hs1) * BasisKet(1, hs=hs2)) ==
                BasisKet(1, hs=hs1) * BasisKet(0, hs=hs2))

    def testOperatorProduct(self):
        hs1 = LocalSpace(1)
        hs2 = LocalSpace(2)
        assert ((Create(hs=hs1) * Destroy(hs=hs1)) *
                (BasisKet(1, hs=hs1) * BasisKet(1, hs=hs2)) ==
                BasisKet(1, hs=hs1) * BasisKet(1, hs=hs2))
        assert ((Create(hs=hs1) * Destroy(hs=hs1) * Destroy(hs=hs1)) *
                (BasisKet(2, hs=hs1)*BasisKet(1, hs=hs2)) ==
                sqrt(2) * BasisKet(1, hs=hs1) * BasisKet(1, hs=hs2))
        assert ((Create(hs=hs1) * Destroy(hs=hs1) * Destroy(hs=hs1)) *
                BasisKet(2, hs=hs1) ==
                sqrt(2) * BasisKet(1, hs=hs1))
        assert ((Create(hs=hs1) * Destroy(hs=hs1)) * BasisKet(1, hs=hs1) ==
                BasisKet(1, hs=hs1))
        assert (
            (Create(hs=hs1) * Destroy(hs=hs1)) * BasisKet(0, hs=hs1) ==
            ZeroKet)


def test_expand_ketbra():
    """Test expansion of KetBra"""
    hs = LocalSpace('0', basis=('0', '1'))
    expr = KetBra(
        KetPlus(BasisKet('0', hs=hs), BasisKet('1', hs=hs)),
        KetPlus(BasisKet('0', hs=hs), BasisKet('1', hs=hs)))
    with temporary_rules(KetBra, clear=True):
        expr_expand = expr.expand()
    assert expr_expand == OperatorPlus(
        KetBra(BasisKet('0', hs=hs), BasisKet('0', hs=hs)),
        KetBra(BasisKet('0', hs=hs), BasisKet('1', hs=hs)),
        KetBra(BasisKet('1', hs=hs), BasisKet('0', hs=hs)),
        KetBra(BasisKet('1', hs=hs), BasisKet('1', hs=hs)))


def test_orthonormality_fock():
    """Test orthonormality of Fock space BasisKets (including symbolic)"""
    hs = LocalSpace('tls', basis=('g', 'e'))
    i = IdxSym('i')
    j = IdxSym('j')
    ket_0 = BasisKet(0, hs=hs)
    bra_0 = ket_0.dag()
    ket_1 = BasisKet(1, hs=hs)
    ket_g = BasisKet('g', hs=hs)
    bra_g = ket_g.dag()
    ket_e = BasisKet('e', hs=hs)
    ket_i = BasisKet(FockIndex(i), hs=hs)
    ket_j = BasisKet(FockIndex(j), hs=hs)
    bra_i = ket_i.dag()
    ket_i_lb = BasisKet(FockLabel(i, hs=hs), hs=hs)
    ket_j_lb = BasisKet(FockLabel(j, hs=hs), hs=hs)
    bra_i_lb = ket_i_lb.dag()

    assert bra_0 * ket_1 == Zero
    assert bra_0 * ket_0 == One

    assert bra_g * ket_g == One
    assert bra_g * ket_e == Zero
    assert bra_0 * ket_g == One
    assert bra_0 * ket_e == Zero
    assert bra_g * ket_0 == One
    assert bra_g * ket_1 == Zero

    delta_ij = KroneckerDelta(i, j)
    delta_i0 = KroneckerDelta(i, 0)
    delta_0j = KroneckerDelta(0, j)
    assert bra_i * ket_j == delta_ij
    assert bra_i * ket_0 == delta_i0
    assert bra_0 * ket_j == delta_0j
    assert bra_i * ket_g == delta_i0
    assert bra_g * ket_j == delta_0j
    assert delta_ij.substitute({i: 0, j: 0}) == One
    assert delta_ij.substitute({i: 0, j: 1}) == Zero
    assert delta_i0.substitute({i: 0}) == One
    assert delta_i0.substitute({i: 1}) == Zero

    delta_ij = KroneckerDelta(i, j)
    delta_ig = KroneckerDelta(i, 0)
    delta_gj = KroneckerDelta(0, j)
    assert bra_i_lb * ket_j_lb == delta_ij
    assert bra_i_lb * ket_0 == delta_ig
    assert bra_0 * ket_j_lb == delta_gj
    assert bra_i_lb * ket_g == delta_ig
    assert bra_g * ket_j_lb == delta_gj
    assert delta_ij.substitute({i: 0, j: 0}) == One
    assert delta_ij.substitute({i: 0, j: 1}) == Zero
    assert delta_ig.substitute({i: 0}) == One
    assert delta_ig.substitute({i: 1}) == Zero


def test_orthonormality_spin():
    hs = SpinSpace('s', spin='1/2')
    i = IdxSym('i')
    j = IdxSym('j')
    ket_dn = SpinBasisKet(-1, 2, hs=hs)
    ket_up = SpinBasisKet(1, 2, hs=hs)
    bra_dn = ket_dn.dag()
    ket_i = BasisKet(SpinIndex(i/2, hs), hs=hs)
    bra_i = ket_i.dag()
    ket_j = BasisKet(SpinIndex(j/2, hs), hs=hs)

    assert bra_dn * ket_dn == One
    assert bra_dn * ket_up == Zero

    delta_ij = KroneckerDelta(i, j, simplify=False)
    delta_i_dn = KroneckerDelta(i, -1, simplify=False)
    delta_dn_j = KroneckerDelta(-1, j, simplify=False)

    assert bra_i * ket_j == delta_ij
    assert bra_i * ket_dn == delta_i_dn
    assert bra_dn * ket_j == delta_dn_j
    assert delta_ij.substitute({i: 0, j: 0}) == One
    assert delta_ij.substitute({i: 0, j: 1}) == Zero


def test_indexed_local_sigma():
    """Test that brakets involving indexed symbols evaluate to Kronecker
    deltas"""
    hs = LocalSpace('tls', basis=('g', 'e'))
    i = IdxSym('i')
    j = IdxSym('j')
    ket_i = BasisKet(FockIndex(i), hs=hs)
    ket_j = BasisKet(FockIndex(j), hs=hs)

    expr = LocalSigma('g', 'e', hs=hs) * ket_i
    expected = KroneckerDelta(i, 1) * BasisKet('g', hs=hs)
    assert expr == expected
    assert expr == LocalSigma(0, 1, hs=hs) * ket_i

    braopket = BraKet(
        ket_i, OperatorTimesKet(
            (LocalSigma('g', 'e', hs=hs) + LocalSigma('e', 'g', hs=hs)),
            ket_j))
    expr = braopket.expand()
    assert expr == (
        KroneckerDelta(i, 0) * KroneckerDelta(1, j) +
        KroneckerDelta(i, 1) * KroneckerDelta(0, j))


def eval_lb(expr, mapping):
    """Evaluate symbolic labels with the given mapping"""
    return _apply_rules(expr, rules=[(
        wc('label', head=SymbolicLabelBase),
        lambda label: label.substitute(mapping))])


def test_ket_symbolic_labels():
    """Test that we can instantiate Kets with symbolic labels"""
    i = IdxSym('i')
    j = IdxSym('j')
    hs0 = LocalSpace(0)
    hs1 = LocalSpace(1)
    Psi = IndexedBase('Psi')

    assert (
        eval_lb(BasisKet(FockIndex(2 * i), hs=hs0), {i: 2}) ==
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
        eval_lb(KetSymbol(StrLabel(2 * i), hs=hs0), {i: 2}) ==
        KetSymbol("4", hs=hs0))
    with pytest.raises(TypeError) as exc_info:
        eval_lb(KetSymbol(FockIndex(2 * i), hs=hs0), {i: 2})
    assert "type of label must be str" in str(exc_info.value)

    assert StrLabel(Psi[i, j]).substitute({i: 'i', j: 'j'}) == 'Psi_ij'
    assert(
        eval_lb(
            KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1), {i: 'i', j: 'j'}) ==
        KetSymbol("Psi_ij", hs=hs0*hs1))
    assert(
        eval_lb(
            KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1), {i: 1, j: 2}) ==
        KetSymbol("Psi_12", hs=hs0*hs1))

    assert (
        eval_lb(
            LocalSigma(FockIndex(i), FockIndex(j), hs=hs0), {i: 1, j: 2}) ==
        LocalSigma(1, 2, hs=hs0))
    assert (
        BasisKet(FockIndex(i), hs=hs0) * BasisKet(FockIndex(j), hs=hs0).dag() ==
        LocalSigma(FockIndex(i), FockIndex(j), hs=hs0))


def test_coherent_state_to_fock_representation():
    """Test the representation of a coherent state in the Fock basis"""
    alpha = symbols('alpha')

    expr1 = CoherentStateKet(alpha, hs=1).to_fock_representation()
    expr2 = CoherentStateKet(alpha, hs=1).to_fock_representation(max_terms=10)
    expr3 = CoherentStateKet(alpha, hs=1).to_fock_representation(
        index_symbol='i')
    expr4 = CoherentStateKet(alpha, hs=1).to_fock_representation(
        index_symbol=IdxSym('m', positive=True))

    assert (
        expr1.term.ranges[0] ==
        IndexOverFockSpace(IdxSym('n'), LocalSpace('1')))
    assert (
        expr2.term.ranges[0] ==
        IndexOverRange(IdxSym('n', integer=True), 0, 9))
    assert (
        expr3.term.ranges[0] ==
        IndexOverFockSpace(IdxSym('i'), LocalSpace('1')))
    assert (
        expr4.term.ranges[0] ==
        IndexOverFockSpace(IdxSym('m', positive=True), LocalSpace('1')))

    for expr in (expr1, expr2):
        assert expr.coeff == exp(-alpha*alpha.conjugate()/2)
        sum = expr.term
        assert len(sum.ranges) == 1
        n = sum.ranges[0].index_symbol
        assert sum.term.coeff == alpha**n/sqrt(factorial(n))
        assert (
            sum.term.term ==
            BasisKet(FockIndex(IdxSym('n')), hs=LocalSpace('1')))


def test_scalar_times_bra():
    """Test that multiplication of a scalar with a bra is handled correctly"""
    alpha_sym = symbols('alpha')
    alpha = ScalarValue(alpha_sym)
    ket = KetSymbol('Psi', hs=0)
    bra = ket.bra

    # first, let's try the ket case, just to establish a working baseline
    expr = alpha * ket
    assert expr == ScalarTimesKet(alpha, ket)
    assert expr == alpha_sym * ket
    assert isinstance((alpha_sym * ket).coeff, ScalarValue)
    assert expr == ket * alpha
    assert expr == ket * alpha_sym

    # now, the bra
    expr = alpha * bra
    assert expr == Bra(ScalarTimesKet(alpha.conjugate(), ket))
    assert expr == alpha_sym * bra
    assert isinstance((alpha_sym * bra).ket.coeff, ScalarValue)
    assert expr == bra * alpha
    assert expr == bra * alpha_sym


def test_disallow_inner_bra():
    """Test that it is not possible to instantiate a State Opereration that has
    a Bra as an operator: we accept Bra to be at the root of the expression
    tree"""
    alpha = symbols('alpha')
    A = OperatorSymbol('A', hs=0)
    ket1 = KetSymbol('Psi_1', hs=0)
    ket2 = KetSymbol('Psi_1', hs=0)
    bra1 = Bra(ket1)
    bra2 = Bra(ket2)
    bra1_hs1 = Bra(KetSymbol('Psi_1', hs=1))

    with pytest.raises(TypeError) as exc_info:
        KetPlus(bra1, bra2)
    assert "must be Kets" in str(exc_info.value)
    assert isinstance(KetPlus.create(bra1, bra2), Bra)

    with pytest.raises(TypeError) as exc_info:
        TensorKet(bra1, bra1_hs1)
    assert "must be Kets" in str(exc_info.value)
    assert isinstance(TensorKet.create(bra1, bra1_hs1), Bra)

    with pytest.raises(TypeError) as exc_info:
        ScalarTimesKet(alpha, bra1)
    assert "must be Kets" in str(exc_info.value)
    assert isinstance(ScalarTimesKet.create(alpha, bra1), Bra)

    with pytest.raises(TypeError) as exc_info:
        OperatorTimesKet(A, bra1)
    assert "must be Kets" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        OperatorTimesKet(bra1, A)
    assert "must be Kets" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        BraKet(bra1, ket2)
    assert "must be Kets" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        KetBra(ket1, bra2)
    assert "must be Kets" in str(exc_info.value)

    i = IdxSym('i')
    Psi = IndexedBase('Psi')
    psi_i = KetSymbol(StrLabel(Psi[i]), hs=0)
    with pytest.raises(TypeError) as exc_info:
        KetIndexedSum(Bra(psi_i), IndexOverFockSpace(i, hs=0))
    assert "must be Kets" in str(exc_info.value)
    assert isinstance(
        KetIndexedSum.create(Bra(psi_i), IndexOverFockSpace(i, hs=0)),
        Bra)
