from functools import partial
from textwrap import dedent

import pytest

from sympy import symbols, sqrt, exp, I, Idx, IndexedBase

from qnet import (
    CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
    Feedback, SeriesInverse, circuit_identity as cid, Beamsplitter,
    OperatorSymbol, IdentityOperator, ZeroOperator, Create, Destroy, Jz, Jplus,
    Jminus, Phase, Displace, Squeeze, LocalSigma, LocalProjector, tr, Adjoint,
    PseudoInverse, NullSpaceProjector, Commutator, OperatorTimes,
    OperatorPlusMinusCC, LocalSpace, TrivialSpace, FullSpace, Matrix,
    KetSymbol, ZeroKet, TrivialKet, BasisKet, CoherentStateKet, UnequalSpaces,
    Bra, OverlappingSpaces, SpaceTooLargeError, BraKet, KetBra,
    SuperOperatorSymbol, IdentitySuperOperator, ZeroSuperOperator,
    SuperAdjoint, SPre, SPost, SuperOperatorTimesOperator, FockIndex, StrLabel,
    IdxSym, ascii, ScalarValue, ScalarExpression, QuantumDerivative, Scalar,
    SpinSpace, Eq)


def test_ascii_scalar():
    """Test rendering of scalar values"""
    assert ascii(2) == ascii(ScalarValue(2)) == '2'
    ascii.printer.cache = {}
    # we always want 2.0 to be printed as '2'. Without this normalization, the
    # state of the cache might introduce non-reproducible behavior, as 2==2.0
    assert ascii(2.0) == ascii(ScalarValue(2.0)) == '2'
    assert ascii(1j) == ascii(ScalarValue(1j)) == '1j'
    assert ascii('foo') == 'foo'

    i = IdxSym('i')
    alpha = IndexedBase('alpha')
    assert ascii(i) == ascii(ScalarValue(i)) == 'i'
    assert ascii(alpha[i]) == ascii(ScalarValue(alpha[i])) == 'alpha_i'


def test_ascii_circuit_elements():
    """Test the ascii representation of "atomic" circuit algebra elements"""
    alpha, t = symbols('alpha, t')
    theta = symbols('theta', positive=True)
    assert ascii(CircuitSymbol("C", cdim=2)) == 'C'
    assert ascii(CircuitSymbol("C_1", cdim=2)) == 'C_1'
    assert ascii(CircuitSymbol("Xi_2", cdim=2)) == 'Xi_2'
    assert ascii(CircuitSymbol("Xi_full", cdim=2)) == 'Xi_full'
    assert ascii(CircuitSymbol("C", alpha, t, cdim=2)) == 'C(alpha, t)'
    with pytest.raises(ValueError):
        CircuitSymbol(r'\Xi^2', cdim=2)
    assert ascii(CIdentity) == 'CIdentity'
    assert ascii(cid(4)) == 'cid(4)'
    assert ascii(CircuitZero) == 'CircuitZero'
    assert ascii(Beamsplitter()) == 'BS(pi/4)'
    assert ascii(Beamsplitter(mixing_angle=theta)) == 'BS(theta)'
    assert ascii(Beamsplitter(label='BS1')) == 'BS1(pi/4)'


def test_ascii_circuit_operations():
    """Test the ascii representation of circuit algebra operations"""
    A = CircuitSymbol("A_test", cdim=2)
    B = CircuitSymbol("B_test", cdim=2)
    C = CircuitSymbol("C_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    perm = CPermutation.create((2, 1, 0, 3))

    assert ascii(A << B << C) == "A_test << B_test << C_test"
    assert ascii(A + B + C) == "A_test + B_test + C_test"
    assert ascii(A << (beta + gamma)) == "A_test << (beta + gamma)"
    assert ascii(A + (B << C)) == "A_test + (B_test << C_test)"
    assert ascii(perm) == "Perm(2, 1, 0, 3)"
    assert (ascii(SeriesProduct(perm, (A+B))) ==
            "Perm(2, 1, 0, 3) << (A_test + B_test)")
    assert (ascii(Feedback((A+B), out_port=3, in_port=0)) ==
            "[A_test + B_test]_{3->0}")
    assert ascii(SeriesInverse(A+B)) == "[A_test + B_test]^{-1}"


def test_ascii_hilbert_elements():
    """Test the ascii representation of "atomic" Hilbert space algebra
    elements"""
    assert ascii(LocalSpace(1)) == 'H_1'
    assert ascii(LocalSpace(1, dimension=2)) == 'H_1'
    assert ascii(LocalSpace(1, basis=('g', 'e'))) == 'H_1'
    assert ascii(LocalSpace('local')) == 'H_local'
    assert ascii(LocalSpace('kappa')) == 'H_kappa'
    with pytest.raises(ValueError):
        LocalSpace(r'\kappa')
    assert ascii(TrivialSpace) == 'H_null'
    assert ascii(FullSpace) == 'H_total'
    assert ascii(LocalSpace(StrLabel(IdxSym('i')))) == 'H_i'


def test_ascii_hilbert_operations():
    """Test the ascii representation of Hilbert space algebra operations"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    assert ascii(H1 * H2) == 'H_1 * H_2'


def test_ascii_matrix():
    """Test ascii representation of the Matrix class"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    assert (ascii(Matrix([[A, B], [C, D]])) ==
            '[[A^(1), B^(1)], [C^(1), D^(1)]]')
    assert (ascii(Matrix([A, B, C, D])) ==
            '[[A^(1)], [B^(1)], [C^(1)], [D^(1)]]')
    assert (ascii(Matrix([[A, B, C, D]])) ==
            '[[A^(1), B^(1), C^(1), D^(1)]]')
    assert ascii(Matrix([[0, 1], [-1, 0]])) == '[[0, 1], [-1, 0]]'
    assert ascii(Matrix([[], []])) == '[[], []]'
    assert ascii(Matrix([])) == '[[], []]'


def test_ascii_equation():
    """Test printing of the Eq class"""
    eq_1 = Eq(
        lhs=OperatorSymbol('H', hs=0),
        rhs=Create(hs=0) * Destroy(hs=0))
    eq = (
        eq_1
        .apply_to_lhs(lambda expr: expr + 1, cont=True)
        .apply_to_rhs(lambda expr: expr + 1, cont=True)
        .apply_to_rhs(lambda expr: expr**2, cont=True, tag=3)
        .apply(lambda expr: expr + 1, cont=True, tag=4)
        .apply_mtd_to_rhs('expand', cont=True)
        .apply_to_lhs(lambda expr: expr**2, cont=True, tag=5)
        .apply_mtd('expand', cont=True)
        .apply_to_lhs(lambda expr: expr**2, cont=True, tag=6)
        .apply_mtd_to_lhs('expand', cont=True)
        .apply_to_rhs(lambda expr: expr + 1, cont=True)
    )
    assert ascii(eq_1) == 'H^(0) = a^(0)H * a^(0)'
    assert ascii(eq_1.set_tag(1)) == 'H^(0) = a^(0)H * a^(0)    (1)'
    assert ascii(eq, show_hs_label=False).strip() == (r'''
                                                       H = a^H * a
                                                   1 + H = a^H * a
                                                         = 1 + a^H * a
                                                         = (1 + a^H * a) * (1 + a^H * a)          (3)
                                                   2 + H = 1 + (1 + a^H * a) * (1 + a^H * a)      (4)
                                                         = 2 + a^H * a^H * a * a + 3 * a^H * a
                                       (2 + H) * (2 + H) = 2 + a^H * a^H * a * a + 3 * a^H * a    (5)
                                       4 + 4 * H + H * H = 2 + a^H * a^H * a * a + 3 * a^H * a
               (4 + 4 * H + H * H) * (4 + 4 * H + H * H) = 2 + a^H * a^H * a * a + 3 * a^H * a    (6)
16 + 32 * H + H * H * H * H + 8 * H * H * H + 24 * H * H = 2 + a^H * a^H * a * a + 3 * a^H * a
                                                         = 3 + a^H * a^H * a * a + 3 * a^H * a
    '''.strip())


def test_ascii_operator_elements():
    """Test the ascii representation of "atomic" operator algebra elements"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    alpha, beta = symbols('alpha, beta')
    assert ascii(OperatorSymbol("A", hs=hs1)) == 'A^(q1)'
    A_1 = OperatorSymbol("A_1", hs=1)
    assert ascii(A_1, show_hs_label='subscript') == 'A_1,(1)'
    assert ascii(OperatorSymbol("A", hs=hs1), show_hs_label=False) == 'A'
    assert ascii(OperatorSymbol("A_1", hs=hs1*hs2)) == 'A_1^(q1*q2)'
    assert ascii(OperatorSymbol("Xi_2", hs=('q1', 'q2'))) == 'Xi_2^(q1*q2)'
    assert ascii(OperatorSymbol("Xi_full", hs=1)) == 'Xi_full^(1)'
    assert ascii(OperatorSymbol("Xi", alpha, beta, hs=1)) == (
        'Xi^(1)(alpha, beta)')
    with pytest.raises(ValueError):
        OperatorSymbol(r'\Xi^2', hs='a')
    assert ascii(IdentityOperator) == "1"
    assert ascii(ZeroOperator) == "0"
    assert ascii(Create(hs=1)) == "a^(1)H"
    assert ascii(Create(hs=1), show_hs_label=False) == "a^H"
    assert ascii(Create(hs=1), show_hs_label='subscript') == "a_(1)^H"
    assert ascii(Destroy(hs=1)) == "a^(1)"
    fock1 = LocalSpace(
       1, local_identifiers={'Create': 'b', 'Destroy': 'b', 'Phase': 'Ph'})
    spin1 = SpinSpace(
       1, spin=1, local_identifiers={'Jz': 'Z', 'Jplus': 'Jp', 'Jminus': 'Jm'})
    assert ascii(Create(hs=fock1)) == "b^(1)H"
    assert ascii(Destroy(hs=fock1)) == "b^(1)"
    assert ascii(Jz(hs=SpinSpace(1, spin=1))) == "J_z^(1)"
    assert ascii(Jz(hs=spin1)) == "Z^(1)"
    assert ascii(Jplus(hs=spin1)) == "Jp^(1)"
    assert ascii(Jminus(hs=spin1)) == "Jm^(1)"
    assert ascii(Phase(0.5, hs=1)) == 'Phase^(1)(0.5)'
    assert ascii(Phase(0.5, hs=fock1)) == 'Ph^(1)(0.5)'
    assert ascii(Displace(0.5, hs=1)) == 'D^(1)(0.5)'
    assert ascii(Squeeze(0.5, hs=1)) == 'Squeeze^(1)(0.5)'
    hs_tls = LocalSpace('1', basis=('g', 'e'))
    sig_e_g = LocalSigma('e', 'g', hs=hs_tls)
    assert ascii(sig_e_g) == '|e><g|^(1)'
    assert ascii(sig_e_g, sig_as_ketbra=False) == 'sigma_e,g^(1)'
    sig_e_e = LocalProjector('e', hs=hs_tls)
    assert ascii(sig_e_e, sig_as_ketbra=False) == 'Pi_e^(1)'
    assert (
        ascii(BasisKet(0, hs=1) * BasisKet(0, hs=2) * BasisKet(0, hs=3)) ==
        '|0,0,0>^(1*2*3)')
    assert (
        ascii(BasisKet(0, hs=hs1) * BasisKet(0, hs=hs2)) ==
        '|00>^(q1*q2)')
    assert (
        ascii(
            BasisKet(0, hs=LocalSpace(0, dimension=20)) *
            BasisKet(0, hs=LocalSpace(1, dimension=20))) ==
        '|0,0>^(0*1)')


def test_ascii_operator_operations():
    """Test the ascii representation of operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    D = OperatorSymbol("D", hs=hs1)
    psi = KetSymbol('Psi', hs=hs1)
    gamma = symbols('gamma', positive=True)
    assert ascii(A + B) == 'A^(q_1) + B^(q_1)'
    assert ascii(A * B) == 'A^(q_1) * B^(q_1)'
    assert ascii(A * C) == 'A^(q_1) * C^(q_2)'
    assert ascii(A * (B + D)) == 'A^(q_1) * (B^(q_1) + D^(q_1))'
    assert ascii(A * (B - D)) == 'A^(q_1) * (B^(q_1) - D^(q_1))'
    assert (
       ascii((A + B) * (-2 * B - D)) ==
       '(A^(q_1) + B^(q_1)) * (-D^(q_1) - 2 * B^(q_1))')
    assert ascii(OperatorTimes(A, -B)) == 'A^(q_1) * (-B^(q_1))'
    assert ascii(OperatorTimes(A, -B), show_hs_label=False) == 'A * (-B)'
    assert ascii(2 * A) == '2 * A^(q_1)'
    assert ascii(2j * A) == '2j * A^(q_1)'
    assert ascii((1+2j) * A) == '(1+2j) * A^(q_1)'
    assert ascii(gamma**2 * A) == 'gamma**2 * A^(q_1)'
    assert ascii(-gamma**2/2 * A) == '-gamma**2/2 * A^(q_1)'
    assert ascii(tr(A * C, over_space=hs2)) == 'tr_(q_2)[C^(q_2)] * A^(q_1)'
    expr = A + OperatorPlusMinusCC(B * D)
    assert ascii(expr, show_hs_label=False) == 'A + (B * D + c.c.)'
    expr = A + OperatorPlusMinusCC(B + D)
    assert ascii(expr, show_hs_label=False) == 'A + (B + D + c.c.)'
    expr = A * OperatorPlusMinusCC(B * D)
    assert ascii(expr, show_hs_label=False) == 'A * (B * D + c.c.)'
    assert ascii(Adjoint(A)) == 'A^(q_1)H'
    assert ascii(Adjoint(Create(hs=1))) == 'a^(1)'
    assert ascii(Adjoint(A + B)) == '(A^(q_1) + B^(q_1))^H'
    assert ascii(PseudoInverse(A)) == '(A^(q_1))^+'
    assert ascii(NullSpaceProjector(A)) == 'P_Ker(A^(q_1))'
    assert ascii(A - B) == 'A^(q_1) - B^(q_1)'
    assert ascii(A - B + C) == 'A^(q_1) - B^(q_1) + C^(q_2)'
    expr = 2 * A - sqrt(gamma) * (B + C)
    assert ascii(expr) == '2 * A^(q_1) - sqrt(gamma) * (B^(q_1) + C^(q_2))'
    assert ascii(Commutator(A, B)) == r'[A^(q_1), B^(q_1)]'
    expr = (Commutator(A, B) * psi).dag()
    assert ascii(expr, show_hs_label=False) == r'<Psi| [A, B]^H'


def test_ascii_ket_elements():
    """Test the ascii representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    alpha, beta = symbols('alpha, beta')
    assert ascii(KetSymbol('Psi', hs=hs1)) == '|Psi>^(q1)'
    psi = KetSymbol('Psi', hs=1)
    assert ascii(psi) == '|Psi>^(1)'
    assert ascii(KetSymbol('Psi', alpha, beta, hs=1)) == (
        '|Psi(alpha, beta)>^(1)')
    assert ascii(psi, show_hs_label='subscript') == '|Psi>_(1)'
    assert ascii(psi, show_hs_label=False) == '|Psi>'
    assert ascii(KetSymbol('Psi', hs=(1, 2))) == '|Psi>^(1*2)'
    assert ascii(KetSymbol('Psi', hs=hs1*hs2)) == '|Psi>^(q1*q2)'
    with pytest.raises(ValueError):
        KetSymbol(r'\Psi', hs=hs1)
    assert ascii(KetSymbol('Psi', hs=1)) == '|Psi>^(1)'
    assert ascii(KetSymbol('Psi', hs=hs1*hs2)) == '|Psi>^(q1*q2)'
    assert ascii(ZeroKet) == '0'
    assert ascii(TrivialKet) == '1'
    assert ascii(BasisKet('e', hs=hs1)) == '|e>^(q1)'
    assert ascii(BasisKet(1, hs=1)) == '|1>^(1)'
    assert ascii(BasisKet(1, hs=hs1)) == '|e>^(q1)'
    with pytest.raises(ValueError):
        BasisKet('1', hs=hs1)
    assert ascii(CoherentStateKet(2.0, hs=1)) == '|alpha=2>^(1)'
    assert ascii(CoherentStateKet(2.1, hs=1)) == '|alpha=2.1>^(1)'


def test_ascii_symbolic_labels():
    """Test ascii representation of symbols with symbolic labels"""
    i = IdxSym('i')
    j = IdxSym('j')
    hs0 = LocalSpace(0)
    hs1 = LocalSpace(1)
    Psi = IndexedBase('Psi')
    assert ascii(BasisKet(FockIndex(2 * i), hs=hs0)) == '|2*i>^(0)'
    assert ascii(KetSymbol(StrLabel(2 * i), hs=hs0)) == '|2*i>^(0)'
    assert (
        ascii(KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1)) == '|Psi_ij>^(0*1)')
    expr = BasisKet(FockIndex(i), hs=hs0) * BasisKet(FockIndex(j), hs=hs1)
    assert ascii(expr) == '|i,j>^(0*1)'
    assert ascii(Bra(BasisKet(FockIndex(2 * i), hs=hs0))) == '<2*i|^(0)'
    assert (
        ascii(LocalSigma(FockIndex(i), FockIndex(j), hs=hs0)) == '|i><j|^(0)')
    expr = CoherentStateKet(symbols('alpha'), hs=1).to_fock_representation()
    assert (
        ascii(expr) ==
        'exp(-alpha*conjugate(alpha)/2) * '
        '(Sum_{n in H_1} alpha**n/sqrt(n!) * |n>^(1))')

    tls = SpinSpace(label='s', spin='1/2', basis=('down', 'up'))
    Sig = IndexedBase('sigma')
    n = IdxSym('n')
    Sig_n = OperatorSymbol(StrLabel(Sig[n]), hs=tls)
    assert ascii(Sig_n, show_hs_label=False) == 'sigma_n'


def test_ascii_bra_elements():
    """Test the ascii representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    bra = Bra(KetSymbol('Psi', hs=1))
    alpha, beta = symbols('alpha, beta')
    assert ascii(Bra(KetSymbol('Psi', hs=hs1))) == '<Psi|^(q1)'
    assert ascii(bra) == '<Psi|^(1)'
    assert ascii(bra, show_hs_label=False) == '<Psi|'
    assert ascii(bra, show_hs_label='subscript') == '<Psi|_(1)'
    assert ascii(Bra(KetSymbol('Psi', alpha, beta, hs=hs1))) == (
        '<Psi(alpha, beta)|^(q1)')
    assert ascii(Bra(KetSymbol('Psi', hs=(1, 2)))) == '<Psi|^(1*2)'
    assert ascii(Bra(KetSymbol('Psi', hs=hs1*hs2))) == '<Psi|^(q1*q2)'
    assert ascii(KetSymbol('Psi', hs=1).dag()) == '<Psi|^(1)'
    assert ascii(Bra(ZeroKet)) == '0'
    assert ascii(Bra(TrivialKet)) == '1'
    assert ascii(BasisKet('e', hs=hs1).adjoint()) == '<e|^(q1)'
    assert ascii(BasisKet(1, hs=1).adjoint()) == '<1|^(1)'
    assert ascii(CoherentStateKet(2.0, hs=1).dag()) == '<alpha=2|^(1)'
    assert ascii(CoherentStateKet(2.1, hs=1).dag()) == '<alpha=2.1|^(1)'
    assert ascii(CoherentStateKet(0.5j, hs=1).dag()) == '<alpha=0.5j|^(1)'


def test_ascii_ket_operations():
    """Test the ascii representation of ket operations"""
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi3 = KetSymbol("Psi_3", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    alpha = symbols('alpha')
    beta = symbols('beta')
    phase = exp(-I * gamma)
    i = IdxSym('i')
    assert ascii(psi1 + psi2) == '|Psi_1>^(q_1) + |Psi_2>^(q_1)'
    assert (ascii(psi1 - psi2 + psi3) ==
            '|Psi_1>^(q_1) - |Psi_2>^(q_1) + |Psi_3>^(q_1)')
    with pytest.raises(UnequalSpaces):
        psi1 + phi
    with pytest.raises(AttributeError):
        (psi1 * phi).label
    assert ascii(psi1 * phi) == '|Psi_1>^(q_1) * |Phi>^(q_2)'
    with pytest.raises(OverlappingSpaces):
        psi1 * psi2
    assert ascii(phase * psi1) == 'exp(-I*gamma) * |Psi_1>^(q_1)'
    assert (
        ascii((alpha + 1) * KetSymbol('Psi', hs=0)) ==
        '(alpha + 1) * |Psi>^(0)')
    assert ascii(A * psi1) == 'A_0^(q_1) |Psi_1>^(q_1)'
    with pytest.raises(SpaceTooLargeError):
        A * phi
    assert ascii(BraKet(psi1, psi2)) == '<Psi_1|Psi_2>^(q_1)'
    expr = BraKet(
        KetSymbol('Psi_1', alpha, hs=hs1), KetSymbol('Psi_2', beta, hs=hs1))
    assert ascii(expr) == '<Psi_1(alpha)|Psi_2(beta)>^(q_1)'
    assert ascii(psi1.dag() * psi2) == '<Psi_1|Psi_2>^(q_1)'
    assert ascii(ket_e1.dag() * ket_e1) == '1'
    assert ascii(ket_g1.dag() * ket_e1) == '0'
    assert ascii(KetBra(psi1, psi2)) == '|Psi_1><Psi_2|^(q_1)'
    expr = KetBra(
        KetSymbol('Psi_1', alpha, hs=hs1), KetSymbol('Psi_2', beta, hs=hs1))
    assert ascii(expr) == '|Psi_1(alpha)><Psi_2(beta)|^(q_1)'
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    assert (ascii(bell1) ==
            '1/sqrt(2) * (|eg>^(q_1*q_2) - I * |ge>^(q_1*q_2))')
    assert (ascii(bell2) ==
            '1/sqrt(2) * (|ee>^(q_1*q_2) - |gg>^(q_1*q_2))')
    expr = BraKet.create(bell1, bell2)
    expected = (
        r'1/2 * (<eg|^(q_1*q_2) + I * <ge|^(q_1*q_2)) * (|ee>^(q_1*q_2) '
        r'- |gg>^(q_1*q_2))')
    assert ascii(expr) == expected
    assert (ascii(KetBra.create(bell1, bell2)) ==
            '1/2 * (|eg>^(q_1*q_2) - I * |ge>^(q_1*q_2))(<ee|^(q_1*q_2) '
            '- <gg|^(q_1*q_2))')
    expr = KetBra(KetSymbol('Psi', hs=0), BasisKet(FockIndex(i), hs=0))
    assert ascii(expr) == "|Psi><i|^(0)"
    expr = KetBra(BasisKet(FockIndex(i), hs=0), KetSymbol('Psi', hs=0))
    assert ascii(expr) == "|i><Psi|^(0)"
    expr = BraKet(KetSymbol('Psi', hs=0), BasisKet(FockIndex(i), hs=0))
    assert ascii(expr) == "<Psi|i>^(0)"
    expr = BraKet(BasisKet(FockIndex(i), hs=0), KetSymbol('Psi', hs=0))
    assert ascii(expr) == "<i|Psi>^(0)"


def test_ascii_bra_operations():
    """Test the ascii representation of bra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi3 = KetSymbol("Psi_3", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    bra_psi1 = KetSymbol("Psi_1", hs=hs1).dag()
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag()
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag()
    bra_psi3 = KetSymbol("Psi_3", hs=hs1).dag()
    bra_phi = KetSymbol("Phi", hs=hs2).dag()
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert ascii((psi1 + psi2).dag()) == '<Psi_1|^(q_1) + <Psi_2|^(q_1)'
    assert ascii(bra_psi1 + bra_psi2) == '<Psi_1|^(q_1) + <Psi_2|^(q_1)'
    assert (ascii((psi1 - psi2 + psi3).dag()) ==
            '<Psi_1|^(q_1) - <Psi_2|^(q_1) + <Psi_3|^(q_1)')
    assert (ascii(bra_psi1 - bra_psi2 + bra_psi3) ==
            '<Psi_1|^(q_1) - <Psi_2|^(q_1) + <Psi_3|^(q_1)')
    assert ascii((psi1 * phi).dag()) == '<Psi_1|^(q_1) * <Phi|^(q_2)'
    assert ascii(bra_psi1 * bra_phi) == '<Psi_1|^(q_1) * <Phi|^(q_2)'
    assert ascii(Bra(phase * psi1)) == 'exp(I*gamma) * <Psi_1|^(q_1)'
    assert ascii((A * psi1).dag()) == '<Psi_1|^(q_1) A_0^(q_1)H'


def test_ascii_sop_elements():
    """Test the ascii representation of "atomic" Superoperators"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    alpha, beta = symbols('alpha, beta')
    assert ascii(SuperOperatorSymbol("A", hs=hs1)) == 'A^(q1)'
    assert ascii(SuperOperatorSymbol("A_1", hs=hs1*hs2)) == 'A_1^(q1*q2)'
    assert (ascii(SuperOperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            'Xi_2^(q1*q2)')
    assert (ascii(SuperOperatorSymbol("Xi", alpha, beta, hs=hs1)) ==
            'Xi^(q1)(alpha, beta)')
    assert ascii(SuperOperatorSymbol("Xi_full", hs=1)) == 'Xi_full^(1)'
    with pytest.raises(ValueError):
        SuperOperatorSymbol(r'\Xi^2', hs='a')
    assert ascii(IdentitySuperOperator) == "1"
    assert ascii(ZeroSuperOperator) == "0"


def test_ascii_sop_operations():
    """Test the ascii representation of super operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = SuperOperatorSymbol("A", hs=hs1)
    B = SuperOperatorSymbol("B", hs=hs1)
    C = SuperOperatorSymbol("C", hs=hs2)
    L = SuperOperatorSymbol("L", hs=1)
    M = SuperOperatorSymbol("M", hs=1)
    A_op = OperatorSymbol("A", hs=1)
    gamma = symbols('gamma', positive=True)
    assert ascii(A + B) == 'A^(q_1) + B^(q_1)'
    assert ascii(A * B) == 'A^(q_1) * B^(q_1)'
    assert ascii(A * C) == 'A^(q_1) * C^(q_2)'
    assert ascii(2 * A) == '2 * A^(q_1)'
    assert ascii(2j * A) == '2j * A^(q_1)'
    assert ascii((1+2j) * A) == '(1+2j) * A^(q_1)'
    assert ascii(gamma**2 * A) == 'gamma**2 * A^(q_1)'
    assert ascii(-gamma**2/2 * A) == '-gamma**2/2 * A^(q_1)'
    assert ascii(SuperAdjoint(A)) == 'A^(q_1)H'
    assert ascii(SuperAdjoint(A + B)) == '(A^(q_1) + B^(q_1))^H'
    assert ascii(A - B) == 'A^(q_1) - B^(q_1)'
    assert ascii(A - B + C) == 'A^(q_1) - B^(q_1) + C^(q_2)'
    assert (ascii(2 * A - sqrt(gamma) * (B + C)) ==
            '2 * A^(q_1) - sqrt(gamma) * (B^(q_1) + C^(q_2))')
    assert ascii(SPre(A_op)) == 'SPre(A^(1))'
    assert ascii(SPost(A_op)) == 'SPost(A^(1))'
    assert (ascii(SuperOperatorTimesOperator(L, A_op)) == 'L^(1)[A^(1)]')
    assert (ascii(SuperOperatorTimesOperator(L, sqrt(gamma) * A_op)) ==
            'L^(1)[sqrt(gamma) * A^(1)]')
    assert (ascii(SuperOperatorTimesOperator((L + 2*M), A_op)) ==
            '(L^(1) + 2 * M^(1))[A^(1)]')


@pytest.fixture
def MyScalarFunc():

    class MyScalarDerivative(QuantumDerivative, Scalar):
        pass

    class ScalarFunc(ScalarExpression):

        def __init__(self, name, *sym_args):
            self._name = name
            self._sym_args = sym_args
            super().__init__(name, *sym_args)

        def _adjoint(self):
            return self

        @property
        def args(self):
            return (self._name, *self._sym_args)

        def _diff(self, sym):
            return MyScalarDerivative(self, derivs={sym: 1})

        def _ascii(self, *args, **kwargs):
            return "%s(%s)" % (
                self._name, ", ".join([ascii(sym) for sym in self._sym_args]))

    return ScalarFunc


def test_ascii_derivative(MyScalarFunc):
    s, s0, t, t0 = symbols('s, s_0, t, t_0', real=True)

    f = partial(MyScalarFunc, "f")
    g = partial(MyScalarFunc, "g")

    expr = f(s, t).diff(s, n=2).diff(t)
    assert ascii(expr) == 'D_s^2 D_t f(s, t)'

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: s0})
    assert ascii(expr) == 'D_s^2 D_t f(s, t) |_(s=s_0)'

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: s0, t: t0})
    assert ascii(expr) == 'D_s^2 D_t f(s, t) |_(s=s_0, t=t_0)'

    D = expr.__class__

    expr = D(f(s, t) + g(s, t), derivs={s: 2, t: 1}, vals={s: s0, t: t0})
    assert ascii(expr) == 'D_s^2 D_t (f(s, t) + g(s, t)) |_(s=s_0, t=t_0)'

    expr = D(2 * f(s, t), derivs={s: 2, t: 1}, vals={s: s0, t: t0})
    assert ascii(expr) == 'D_s^2 D_t (2 * f(s, t)) |_(s=s_0, t=t_0)'

    expr = f(s, t).diff(t) + g(s, t)
    assert ascii(expr) == 'D_t f(s, t) + g(s, t)'

    expr = f(s, t).diff(t) * g(s, t)
    assert ascii(expr) == '(D_t f(s, t)) * g(s, t)'

    expr = (  # nested derivative
        MyScalarFunc("f", s, t)
        .diff(s, n=2)
        .diff(t)
        .evaluate_at({t: t0})
        .diff(t0))
    assert ascii(expr) == 'D_t_0 (D_s^2 D_t f(s, t) |_(t=t_0))'
