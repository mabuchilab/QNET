from textwrap import dedent

import pytest

# We need lots of extra import so that all "srepr" have a context in which they
# can be evaluated
from sympy import (
    symbols, sqrt, exp, I, Float, Pow, Mul, Integer, Symbol, Rational,
    factorial)
from numpy import array, float64, complex128, int64, int32

from qnet import(
    CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
    Concatenation, Feedback, SeriesInverse, OperatorSymbol, IdentityOperator,
    ZeroOperator, Create, Destroy, Jz, Jplus, Jminus, Phase, Displace, Squeeze,
    LocalSigma, tr, Adjoint, PseudoInverse, NullSpaceProjector, OperatorPlus,
    OperatorTimes, ScalarTimesOperator, OperatorTrace, Commutator,
    OperatorIndexedSum, OperatorDerivative, LocalSpace, TrivialSpace,
    FullSpace, ProductSpace, Matrix, KetSymbol, ZeroKet, TrivialKet, BasisKet,
    CoherentStateKet, UnequalSpaces, OperatorTimesKet, Bra, KetPlus,
    ScalarTimesKet, OverlappingSpaces, SpaceTooLargeError, BraKet, KetBra,
    TensorKet, KetIndexedSum, SuperOperatorSymbol, IdentitySuperOperator,
    ZeroSuperOperator, SuperAdjoint, SPre, SPost, SuperOperatorTimesOperator,
    SuperOperatorPlus, SuperOperatorTimes, ScalarTimesSuperOperator, IdxSym,
    FockIndex, IndexOverFockSpace, srepr, ScalarValue, ScalarTimes, One, Zero,
    SpinSpace, Beamsplitter)
from qnet.printing._render_head_repr import render_head_repr


def test_render_head_repr_tuple_list():
    """Test that render_head_repr works for lists and tuples"""
    a, b = symbols('a, b')
    A = OperatorSymbol('A', hs=0)
    B = OperatorSymbol('B', hs=0)

    expr = ((a, 2), (b, 1))
    assert render_head_repr(expr) == "((Symbol('a'), 2), (Symbol('b'), 1))"

    expr = [[a, 2], [b, 1]]
    assert render_head_repr(expr) == "[[Symbol('a'), 2], [Symbol('b'), 1]]"

    expr = [(a, 2), (b, 1)]
    assert render_head_repr(expr) == "[(Symbol('a'), 2), (Symbol('b'), 1)]"

    expr = ([a, 2], [b, 1])
    assert render_head_repr(expr) == "([Symbol('a'), 2], [Symbol('b'), 1])"

    expr = (A, (b, 1))
    assert (
        render_head_repr(expr) ==
        "(OperatorSymbol('A', hs=LocalSpace('0')), (Symbol('b'), 1))")

    expr = [(a, 2), (A, B)]
    assert (
        render_head_repr(expr) ==
        "[(Symbol('a'), 2), (OperatorSymbol('A', hs=LocalSpace('0')), "
        "OperatorSymbol('B', hs=LocalSpace('0')))]")


def test_indented_srepr_tuple_list():
    """Test that indendented srepr of a list or tuple is same as unindented

    Because these occur as kwargs values, they are always printed unindented.
    """
    a, b = symbols('a, b')
    A = OperatorSymbol('A', hs=0)
    B = OperatorSymbol('B', hs=0)
    exprs = [
        ((a, 2), (b, 1)),
        [[a, 2], [b, 1]],
        [(a, 2), (b, 1)],
        ([a, 2], [b, 1]),
        (A, (b, 1)),
        [(a, 2), (A, B)],
    ]
    for expr in exprs:
        assert (
            srepr(expr, indented=True) ==
            srepr(expr) ==
            render_head_repr(expr))


def test_srepr_local_space():
    """Test that the srepr of a LocalSpace is "nice" (as Hilbert spaces a
    have a lot of keyword arguments, which we only want to show if necessary"""
    assert srepr(LocalSpace(1)) == r"LocalSpace('1')"
    assert srepr(LocalSpace(1, dimension=2)) == r"LocalSpace('1', dimension=2)"
    assert (
        srepr(LocalSpace(1, dimension=2, basis=('g', 'e'))) ==
        r"LocalSpace('1', basis=('g', 'e'), dimension=2)")
    assert (
        srepr(LocalSpace(1, basis=('g', 'e'))) ==
        r"LocalSpace('1', basis=('g', 'e'))")
    assert (
        srepr(LocalSpace(1, basis=('g', 'e'), order_index=1)) ==
        r"LocalSpace('1', basis=('g', 'e'), order_index=1)")
    assert (
        srepr(LocalSpace(1, local_identifiers={'Destroy': 'b'})) ==
        r"LocalSpace('1', local_identifiers=(('Destroy', 'b'),))")


def test_srepr_circuit_elements():
    """Test the representation of "atomic" circuit algebra elements"""
    alpha = symbols('alpha')
    assert (
        srepr(CircuitSymbol("C_1", cdim=2)) ==
        "CircuitSymbol('C_1', cdim=2)")
    assert (
        srepr(CircuitSymbol("A", alpha, 0, 2, cdim=2)) ==
        "CircuitSymbol('A', Symbol('alpha'), 0, 2, cdim=2)")
    assert srepr(CIdentity) == r'CIdentity'
    assert srepr(CircuitZero) == r'CircuitZero'
    assert srepr(Beamsplitter()) == r'Beamsplitter()'
    assert srepr(Beamsplitter(label='BS1')) == r"Beamsplitter(label='BS1')"
    assert (
        srepr(Beamsplitter(mixing_angle=alpha)) ==
        r"Beamsplitter(mixing_angle=Symbol('alpha'))")


def test_srepr_idx_sym():
    """Test the representation of IdxSym instances"""
    i = IdxSym('i')
    assert srepr(i) == "IdxSym('i', integer=True)"
    assert srepr(i.prime) == "IdxSym('i', integer=True, primed=1)"
    assert eval(srepr(i)) == i
    assert eval(srepr(i.prime)) == i.prime

    i_pos = IdxSym('i', positive=True)
    assert i != i_pos
    assert srepr(i_pos) == "IdxSym('i', integer=True, positive=True)"
    assert eval(srepr(i_pos)) == i_pos
    assert (
        srepr(i_pos.prime.prime) ==
        "IdxSym('i', integer=True, positive=True, primed=2)")
    assert eval(srepr(i_pos.prime.prime)) == i_pos.prime.prime

    i_nonint = IdxSym('i', integer=False)
    assert i != i_nonint
    assert srepr(i_nonint) == "IdxSym('i', integer=False)"
    assert eval(srepr(i_nonint)) == i_nonint
    assert eval(srepr(i_nonint.prime.prime)) == i_nonint.prime.prime


def test_indented_srepr():
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    A = OperatorSymbol("A", hs=1)
    res = srepr(hs1, indented=True)
    expected = dedent(r'''
        LocalSpace(
            'q_1',
            basis=('g', 'e'))''').strip()
    assert res == expected

    # test that caching doesn't mess up indentation
    expr = OperatorPlus(A, A)
    res = srepr(expr, indented=True)
    expected = dedent(r'''
    OperatorPlus(
        OperatorSymbol(
            'A',
            hs=LocalSpace(
                '1')),
        OperatorSymbol(
            'A',
            hs=LocalSpace(
                '1')))''').strip()
    assert res == expected


@pytest.fixture
def matrix_expr():
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    gamma = symbols('gamma')
    phase = exp(-I * gamma / 2)
    return Matrix([[phase*A, B], [C, phase.conjugate()*D]])


@pytest.fixture
def bell1_expr():
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    return (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)


def test_foreign_srepr(matrix_expr, bell1_expr):
    """Test that srepr also works on sympy/numpy components"""

    res = srepr(matrix_expr)
    expected = (
        "Matrix(array([[ScalarTimesOperator(ScalarValue(exp(Mul(Integer(-1), "
        "Rational(1, 2), I, Symbol('gamma')))), OperatorSymbol('A', "
        "hs=LocalSpace('1'))), OperatorSymbol('B', hs=LocalSpace('1'))], "
        "[OperatorSymbol('C', hs=LocalSpace('1')), "
        "ScalarTimesOperator(ScalarValue(exp(Mul(Rational(1, 2), I, "
        "conjugate(Symbol('gamma'))))), OperatorSymbol('D', "
        "hs=LocalSpace('1')))]], dtype=object))")
    assert res == expected

    res = srepr(matrix_expr, indented=True)
    expected = dedent(r'''
    Matrix(
        array([
            [
                ScalarTimesOperator(
                    ScalarValue(
                        exp(Mul(Integer(-1), Rational(1, 2), I, Symbol('gamma')))),
                    OperatorSymbol(
                        'A',
                        hs=LocalSpace(
                            '1'))),
                OperatorSymbol(
                    'B',
                    hs=LocalSpace(
                        '1'))
            ],
            [
                OperatorSymbol(
                    'C',
                    hs=LocalSpace(
                        '1')),
                ScalarTimesOperator(
                    ScalarValue(
                        exp(Mul(Rational(1, 2), I, conjugate(Symbol('gamma'))))),
                    OperatorSymbol(
                        'D',
                        hs=LocalSpace(
                            '1')))
            ]], dtype=object))''').strip()
    assert res == expected

    expected = (
        "ScalarTimesKet(ScalarValue(Mul(Rational(1, 2), Pow(Integer(2), "
        "Rational(1, 2)))), KetPlus(TensorKet(BasisKet('e', "
        "hs=LocalSpace('q_1', basis=('g', 'e'))), BasisKet('g', "
        "hs=LocalSpace('q_2', basis=('g', 'e')))), "
        "ScalarTimesKet(ScalarValue(Mul(Integer(-1), I)), "
        "TensorKet(BasisKet('g', hs=LocalSpace('q_1', basis=('g', 'e'))), "
        "BasisKet('e', hs=LocalSpace('q_2', basis=('g', 'e')))))))")
    assert srepr(bell1_expr) == expected

    res = srepr(bell1_expr, indented=True)
    expected = dedent(r'''
    ScalarTimesKet(
        ScalarValue(
            Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2)))),
        KetPlus(
            TensorKet(
                BasisKet(
                    'e',
                    hs=LocalSpace(
                        'q_1',
                        basis=('g', 'e'))),
                BasisKet(
                    'g',
                    hs=LocalSpace(
                        'q_2',
                        basis=('g', 'e')))),
            ScalarTimesKet(
                ScalarValue(
                    Mul(Integer(-1), I)),
                TensorKet(
                    BasisKet(
                        'g',
                        hs=LocalSpace(
                            'q_1',
                            basis=('g', 'e'))),
                    BasisKet(
                        'e',
                        hs=LocalSpace(
                            'q_2',
                            basis=('g', 'e')))))))''').strip()
    assert res == expected


def test_cached_srepr(bell1_expr):
    """Test that we can get simplified expressions by passing a cache, and that
    the cache is updated appropriately while printing"""
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)

    cache = {hs1: 'hs1', hs2: 'hs2', 1/sqrt(2): '1/sqrt(2)', -I: '-I'}
    res = srepr(bell1_expr, cache=cache)
    expected = (
        "ScalarTimesKet(1/sqrt(2), KetPlus(TensorKet(BasisKet('e', hs=hs1), "
        "BasisKet('g', hs=hs2)), ScalarTimesKet(-I, "
        "TensorKet(BasisKet('g', hs=hs1), BasisKet('e', hs=hs2)))))")
    assert res == expected

    assert ket_g1 in cache
    assert cache[ket_g1] == "BasisKet('g', hs=hs1)"

    cache = {hs1: 'hs1', hs2: 'hs2', 1/sqrt(2): '1/sqrt(2)', -I: '-I'}
    # note that we *must* use a different cache
    res = srepr(bell1_expr, cache=cache, indented=True)
    expected = dedent(r'''
    ScalarTimesKet(
        1/sqrt(2),
        KetPlus(
            TensorKet(
                BasisKet(
                    'e',
                    hs=hs1),
                BasisKet(
                    'g',
                    hs=hs2)),
            ScalarTimesKet(
                -I,
                TensorKet(
                    BasisKet(
                        'g',
                        hs=hs1),
                    BasisKet(
                        'e',
                        hs=hs2)))))''').strip()
    assert res == expected

    assert ket_g1 in cache
    assert cache[ket_g1] == "BasisKet(\n    'g',\n    hs=hs1)"


def circuit_exprs():
    """Prepare a list of circuit algebra expressions"""
    A = CircuitSymbol("A_test", cdim=2)
    B = CircuitSymbol("B_test", cdim=2)
    C = CircuitSymbol("C_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    perm = CPermutation.create((2, 1, 0, 3))

    return [
        CircuitSymbol("C_1", cdim=2),
        CIdentity,
        CircuitZero,
        Beamsplitter(),
        Beamsplitter(label='BS1'),
        Beamsplitter(label='BS1', mixing_angle=symbols('phi', positive=True)),
        A << B << C,
        A + B + C,
        A << (beta + gamma),
        A + (B << C),
        CircuitSymbol("A", 0, symbols('alpha'), cdim=2),
        perm,
        SeriesProduct(perm, (A+B)),
        Feedback((A+B), out_port=3, in_port=0),
        SeriesInverse(A+B),
    ]


def hilbert_exprs():
    """Prepare a list of Hilbert space algebra expressions"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    return [
        LocalSpace(1),
        LocalSpace(1, dimension=2),
        LocalSpace(1, basis=(r'g', 'e')),
        LocalSpace('kappa'),
        TrivialSpace,
        FullSpace,
        H1 * H2,
    ]


def matrix_exprs():
    """Prepare a list of Matrix expressions"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    return [
        Matrix([[A, B], [C, D]]),
        Matrix([A, B, C, D]),
        Matrix([[A, B, C, D]]),
        Matrix([[0, 1], [-1, 0]]),
        #Matrix([[], []]),  # see issue #8316 in numpy
        #Matrix([]),
    ]


def operator_exprs():
    """Prepare a list of operator algebra expressions"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    a, b = symbols('a, b')
    A_ab = OperatorSymbol("A", a, b, hs=0)
    gamma = symbols('gamma')
    return [
        OperatorSymbol("A", hs=hs1),
        OperatorSymbol("A_1", hs=hs1*hs2),
        OperatorSymbol("A_1", symbols('alpha'), symbols('beta'), hs=hs1*hs2),
        A_ab.diff(a, n=2).diff(b),
        A_ab.diff(a, n=2).diff(b).evaluate_at({a: 0}),
        OperatorSymbol("Xi_2", hs=(r'q1', 'q2')),
        OperatorSymbol("Xi_full", hs=1),
        IdentityOperator,
        ZeroOperator,
        Create(hs=1),
        Create(hs=LocalSpace(1, local_identifiers={'Create': 'b'})),
        Destroy(hs=1),
        Destroy(hs=LocalSpace(1, local_identifiers={'Destroy': 'b'})),
        Jz(hs=SpinSpace(1, spin=1)),
        Jz(hs=SpinSpace(1, spin=1, local_identifiers={'Jz': 'Z'})),
        Jplus(hs=SpinSpace(1, spin=1, local_identifiers={'Jplus': 'Jp'})),
        Jminus(hs=SpinSpace(1, spin=1, local_identifiers={'Jminus': 'Jm'})),
        Phase(0.5, hs=1),
        Phase(0.5, hs=LocalSpace(1, local_identifiers={'PhaseCC': 'Ph'})),
        Displace(0.5, hs=1),
        Squeeze(0.5, hs=1),
        LocalSigma('e', 'g', hs=LocalSpace(1, basis=('g', 'e'))),
        LocalSigma('e', 'e', hs=LocalSpace(1, basis=('g', 'e'))),
        A + B,
        A * B,
        A * C,
        2 * A,
        2j * A,
        (1+2j) * A,
        gamma**2 * A,
        -gamma**2/2 * A,
        tr(A * C, over_space=hs2),
        Adjoint(A),
        Adjoint(A + B),
        PseudoInverse(A),
        NullSpaceProjector(A),
        A - B,
        2 * A - sqrt(gamma) * (B + C),
        Commutator(A, B),
    ]


def state_exprs():
    """Prepare a list of state algebra expressions"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi1_l = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi3 = KetSymbol("Psi_3", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    phi_l = KetSymbol("Phi", hs=hs2)
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma')
    phase = exp(-I * gamma)
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    bra_psi1 = KetSymbol("Psi_1", hs=hs1).dag()
    bra_psi1_l = KetSymbol("Psi_1", hs=hs1).dag()
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag()
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag()
    bra_phi_l = KetSymbol("Phi", hs=hs2).dag()
    return [
        KetSymbol('Psi', hs=hs1),
        KetSymbol('Psi', hs=1),
        KetSymbol('Psi', hs=(1, 2)),
        KetSymbol('Psi', symbols('alpha'), symbols('beta'), hs=(1, 2)),
        KetSymbol('Psi', hs=1),
        ZeroKet,
        TrivialKet,
        BasisKet('e', hs=hs1),
        BasisKet('excited', hs=LocalSpace(1, basis=('ground', 'excited'))),
        BasisKet(1, hs=1),
        CoherentStateKet(2.0, hs=1),
        CoherentStateKet(2.0, hs=1).to_fock_representation(),
        Bra(KetSymbol('Psi', hs=hs1)),
        Bra(KetSymbol('Psi', hs=1)),
        Bra(KetSymbol('Psi', hs=(1, 2))),
        Bra(KetSymbol('Psi', hs=hs1*hs2)),
        KetSymbol('Psi', hs=1).dag(),
        Bra(ZeroKet),
        Bra(TrivialKet),
        BasisKet('e', hs=hs1).adjoint(),
        BasisKet(1, hs=1).adjoint(),
        CoherentStateKet(2.0, hs=1).dag(),
        psi1 + psi2,
        psi1 - psi2 + psi3,
        psi1 * phi,
        psi1_l * phi_l,
        phase * psi1,
        A * psi1,
        BraKet(psi1, psi2),
        ket_e1.dag() * ket_e1,
        ket_g1.dag() * ket_e1,
        KetBra(psi1, psi2),
        bell1,
        BraKet.create(bell1, bell2),
        KetBra.create(bell1, bell2),
        (psi1 + psi2).dag(),
        bra_psi1 + bra_psi2,
        bra_psi1_l * bra_phi_l,
        Bra(phase * psi1),
        (A * psi1).dag(),
    ]


def sop_exprs():
    """Prepare a list of super operator algebra expressions"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    A = SuperOperatorSymbol("A", hs=hs1)
    B = SuperOperatorSymbol("B", hs=hs1)
    C = SuperOperatorSymbol("C", hs=hs2)
    L = SuperOperatorSymbol("L", hs=1)
    M = SuperOperatorSymbol("M", hs=1)
    A_op = OperatorSymbol("A", hs=1)
    gamma = symbols('gamma')
    return [
        SuperOperatorSymbol("A", hs=hs1),
        SuperOperatorSymbol("A_1", hs=hs1*hs2),
        SuperOperatorSymbol("A", symbols('alpha'), symbols('beta'), hs=hs1),
        IdentitySuperOperator,
        ZeroSuperOperator,
        A + B,
        A * B,
        A * C,
        2 * A,
        (1+2j) * A,
        -gamma**2/2 * A,
        SuperAdjoint(A + B),
        2 * A - sqrt(gamma) * (B + C),
        SPre(A_op),
        SPost(A_op),
        SuperOperatorTimesOperator(L, sqrt(gamma) * A_op),
        SuperOperatorTimesOperator((L + 2*M), A_op),
    ]


@pytest.mark.parametrize(
    'expr',
    (circuit_exprs() + hilbert_exprs() + matrix_exprs() + operator_exprs() +
     state_exprs() + sop_exprs()))
def test_self_eval(expr):
    s = srepr(expr)
    assert eval(s) == expr
    s = srepr(expr, indented=True)
    assert eval(s) == expr
