from functools import partial

import pytest

import sympy
from sympy import symbols, sqrt, exp, I, Rational, IndexedBase

from qnet import (
    CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
    Feedback, SeriesInverse, circuit_identity as cid, Beamsplitter,
    OperatorSymbol, IdentityOperator, ZeroOperator, Create, Destroy, Jz, Jplus,
    Jminus, Phase, Displace, Squeeze, LocalSigma, LocalProjector, tr, Adjoint,
    PseudoInverse, NullSpaceProjector, Commutator, LocalSpace, TrivialSpace,
    FullSpace, Matrix, KetSymbol, ZeroKet, TrivialKet, BasisKet,
    CoherentStateKet, UnequalSpaces, ScalarTimesKet, OperatorTimesKet, Bra,
    OverlappingSpaces, SpaceTooLargeError, BraKet, KetBra, SuperOperatorSymbol,
    IdentitySuperOperator, ZeroSuperOperator, SuperAdjoint, SPre, SPost,
    SuperOperatorTimesOperator, FockIndex, StrLabel, IdxSym, unicode,
    QuantumDerivative, Scalar, ScalarExpression, SpinSpace, Eq, SpinSpace)


def test_unicode_scalar():
    """Test rendering of scalar values"""
    assert unicode(2) == '2'
    # we always want 2.0 to be printed as '2'. Without this normalization, the
    # state of the cache might introduce non-reproducible behavior, as 2==2.0
    unicode.printer.cache = {}
    assert 2 == 2.0
    assert unicode(2.0) == '2'   # would be '2.0' without normalization
    assert unicode(1j) == '1j'
    assert I/2 == 0.5j
    assert unicode(I/2) == 'ⅈ/2'
    assert unicode(0.5j) == '0.5j'
    assert unicode(sympy.pi) == 'π'
    assert unicode(sympy.pi/4) == 'π/4'

    i = IdxSym('i')
    alpha = IndexedBase('alpha')
    assert unicode(alpha[i]) == 'α_i'
    assert unicode(alpha[1]) == 'α₁'


def test_unicode_circuit_elements():
    """Test the unicode representation of "atomic" circuit algebra elements"""
    alpha, t = symbols('alpha, t')
    theta = symbols('theta', positive=True)
    assert unicode(CircuitSymbol("C", cdim=2)) == 'C'
    C_1 = CircuitSymbol("C_1", cdim=2)
    assert unicode(C_1) == 'C₁'
    assert unicode(C_1, unicode_sub_super=False) == 'C_1'
    assert unicode(CircuitSymbol("Xi_2", cdim=2)) == 'Ξ₂'
    assert unicode(CircuitSymbol("Xi_full", cdim=2)) == 'Ξ_full'
    assert unicode(CircuitSymbol("C", alpha, t, cdim=2)) == 'C(α, t)'
    assert unicode(CIdentity) == 'CIdentity'
    assert unicode(cid(4)) == 'cid(4)'
    assert unicode(CircuitZero) == 'CircuitZero'
    assert unicode(Beamsplitter()) == 'BS(π/4)'
    assert unicode(Beamsplitter(mixing_angle=theta)) == 'BS(θ)'


def test_unicode_circuit_operations():
    """Test the unicode representation of circuit algebra operations"""
    A = CircuitSymbol("A_test", cdim=2)
    B = CircuitSymbol("B_test", cdim=2)
    C = CircuitSymbol("C_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    perm = CPermutation.create((2, 1, 0, 3))

    assert unicode(A << B << C) == "A_test ◁ B_test ◁ C_test"
    assert unicode(A + B + C) == 'A_test ⊞ B_test ⊞ C_test'
    assert unicode(A << (beta + gamma)) == "A_test ◁ (β ⊞ γ)"
    assert unicode(A + (B << C)) == "A_test ⊞ (B_test ◁ C_test)"
    assert unicode(perm) == "Perm(2, 1, 0, 3)"
    assert (unicode(SeriesProduct(perm, (A+B))) ==
            "Perm(2, 1, 0, 3) ◁ (A_test ⊞ B_test)")
    fb = Feedback((A+B), out_port=3, in_port=0)
    assert (unicode(fb) == "[A_test ⊞ B_test]₃₋₀")
    assert (unicode(fb, unicode_sub_super=False) == "[A_test ⊞ B_test]_3→0")
    assert unicode(SeriesInverse(A+B)) == "[A_test ⊞ B_test]⁻¹"


def test_unicode_hilbert_elements():
    """Test the unicode representation of "atomic" Hilbert space algebra
    elements"""
    assert unicode(LocalSpace(1)) == 'ℌ₁'
    assert unicode(LocalSpace(1, dimension=2)) == 'ℌ₁'
    assert unicode(LocalSpace(1, basis=('g', 'e'))) == 'ℌ₁'
    assert unicode(LocalSpace('local')) == 'ℌ_local'
    assert unicode(LocalSpace('kappa')) == 'ℌ_κ'
    assert unicode(TrivialSpace) == 'ℌ_null'
    assert unicode(FullSpace) == 'ℌ_total'
    assert unicode(LocalSpace(StrLabel(IdxSym('i')))) == 'ℌ_i'


def test_unicode_hilbert_operations():
    """Test the unicode representation of Hilbert space algebra operations"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    assert unicode(H1 * H2) == 'ℌ₁ ⊗ ℌ₂'


def test_unicode_matrix():
    """Test unicode representation of the Matrix class"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    assert (unicode(Matrix([[A, B], [C, D]])) ==
            '[[A\u0302\u207d\xb9\u207e, B\u0302\u207d\xb9\u207e], '
            '[C\u0302\u207d\xb9\u207e, D\u0302\u207d\xb9\u207e]]')
            #  '[[Â⁽¹⁾, B̂⁽¹⁾], [Ĉ⁽¹⁾, D̂⁽¹⁾]]')
    assert (unicode(Matrix([A, B, C, D])) ==
            '[[A\u0302\u207d\xb9\u207e], [B\u0302\u207d\xb9\u207e], '
            '[C\u0302\u207d\xb9\u207e], [D\u0302\u207d\xb9\u207e]]')
            #  '[Â⁽¹⁾], [B̂⁽¹⁾], [Ĉ⁽¹⁾], [D̂⁽¹⁾]]')
    assert unicode(Matrix([[0, 1], [-1, 0]])) == '[[0, 1], [-1, 0]]'
    assert unicode(Matrix([[], []])) == '[[], []]'
    assert unicode(Matrix([])) == '[[], []]'


def test_unicode_equation():
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
    assert unicode(eq_1) == 'Ĥ⁽⁰⁾ = â^(0)† â⁽⁰⁾'
    assert unicode(eq_1.set_tag(1)) == 'Ĥ⁽⁰⁾ = â^(0)† â⁽⁰⁾    (1)'
    unicode_lines = unicode(eq, show_hs_label=False).split("\n")
    expected = [
        '                                     Ĥ = â^† â',
        '                                 𝟙 + Ĥ = â^† â',
        '                                       = 𝟙 + â^† â',
        '                                       = (𝟙 + â^† â) (𝟙 + â^† â)        (3)',
        '                                 2 + Ĥ = 𝟙 + (𝟙 + â^† â) (𝟙 + â^† â)    (4)',
        '                                       = 2 + â^† â^† â â + 3 â^† â',
        '                       (2 + Ĥ) (2 + Ĥ) = 2 + â^† â^† â â + 3 â^† â      (5)',
        '                         4 + 4 Ĥ + Ĥ Ĥ = 2 + â^† â^† â â + 3 â^† â',
        '       (4 + 4 Ĥ + Ĥ Ĥ) (4 + 4 Ĥ + Ĥ Ĥ) = 2 + â^† â^† â â + 3 â^† â      (6)',
        '16 + 32 Ĥ + Ĥ Ĥ Ĥ Ĥ + 8 Ĥ Ĥ Ĥ + 24 Ĥ Ĥ = 2 + â^† â^† â â + 3 â^† â',
        '                                       = 3 + â^† â^† â â + 3 â^† â']
    for i, line in enumerate(unicode_lines):
        assert line == expected[i]


def test_unicode_operator_elements():
    """Test the unicode representation of "atomic" operator algebra elements"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    alpha, beta = symbols('alpha, beta')
    assert unicode(OperatorSymbol("A", hs=hs1)) == 'A\u0302^(q\u2081)'
    #                                               Â^(q₁)
    assert(
        unicode(OperatorSymbol('A', hs=1), show_hs_label='subscript') ==
        'A\u0302\u208d\u2081\u208e')  # Â₍₁₎
    assert unicode(
        OperatorSymbol("A", hs=hs1),
        unicode_op_hats=False, unicode_sub_super=False) == 'A^(q_1)'
    assert (unicode(OperatorSymbol("A_1", hs=hs1*hs2)) ==
            'A\u0302_1^(q\u2081\u2297q\u2082)')  # Â_1^(q₁⊗q₂)
    assert (unicode(OperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            '\u039e\u0302_2^(q\u2081\u2297q\u2082)')  # Ξ̂_2^(q₁⊗q₂)
    assert unicode(OperatorSymbol("Xi", alpha, beta, hs=1)) == (
        'Ξ̂⁽¹⁾(α, β)')
    assert unicode(IdentityOperator) == "𝟙"
    assert unicode(ZeroOperator) == "0"
    assert unicode(Create(hs=1)) == 'a\u0302^(1)\u2020'  # â^(1)†
    assert unicode(Destroy(hs=1)) == 'a\u0302\u207d\xb9\u207e'  # â⁽¹⁾
    assert (
        unicode(Destroy(hs=1), unicode_sub_super=False) == 'a\u0302^(1)')
    assert (
        unicode(Destroy(hs=1), unicode_op_hats=False) == 'a\u207d\xb9\u207e')
    assert (
        unicode(
            Destroy(hs=1), unicode_op_hats=False, unicode_sub_super=False) ==
        'a^(1)')
    assert (unicode(Squeeze(Rational(1, 2), hs=1)) ==
            'Squeeze\u207d\xb9\u207e(1/2)')
    #       Squeeze⁽¹⁾(1/2)
    hs_tls = LocalSpace('1', basis=('g', 'e'))
    sig_e_g = LocalSigma('e', 'g', hs=hs_tls)
    assert unicode(sig_e_g) == '|e⟩⟨g|⁽¹⁾'
    assert unicode(sig_e_g, unicode_sub_super=False) == '|e⟩⟨g|^(1)'
    assert unicode(sig_e_g, show_hs_label=False) == '|e⟩⟨g|'
    assert (
        unicode(sig_e_g, sig_as_ketbra=False) ==
        '\u03c3\u0302_e,g^(1)')  # σ̂_e,g^(1)
    sig_e_e = LocalProjector('e', hs=hs_tls)
    assert unicode(sig_e_e) == '|e⟩⟨e|⁽¹⁾'
    assert (
        unicode(sig_e_e, sig_as_ketbra=False) ==
        '\u03a0\u0302\u2091\u207d\xb9\u207e')  # Π̂ₑ⁽¹⁾
    assert (
        unicode(BasisKet(0, hs=1) * BasisKet(0, hs=2) * BasisKet(0, hs=3)) ==
        '|0,0,0⟩^(1⊗2⊗3)')
    assert (
        unicode(BasisKet(0, hs=hs1) * BasisKet(0, hs=hs2)) ==
        '|00⟩^(q₁⊗q₂)')


def test_unicode_operator_operations():
    """Test the unicode representation of operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    psi = KetSymbol('Psi', hs=hs1)
    gamma = symbols('gamma', positive=True)
    assert unicode(A + B) == 'A\u0302^(q\u2081) + B\u0302^(q\u2081)'
    #                         Â^(q₁) + B̂^(q₁)
    assert unicode(A * B) == 'A\u0302^(q\u2081) B\u0302^(q\u2081)'
    #                         Â^(q₁) B̂^(q₁)
    assert unicode(A * C) == 'A\u0302^(q\u2081) C\u0302^(q\u2082)'
    #                         Â^(q₁) Ĉ^(q₂)
    assert unicode(2 * A) == '2 A\u0302^(q\u2081)'  # 2 Â^(q₁)
    assert unicode(2j * A) == '2j A\u0302^(q\u2081)'
    #                          2j Â^(q₁)
    assert unicode((1+2j) * A) == '(1+2j) A\u0302^(q\u2081)'
    #                              (1+2j) Â^(q₁)
    assert unicode(gamma**2 * A) == '\u03b3\xb2 A\u0302^(q\u2081)'
    #                                γ² Â^(q₁)
    assert unicode(-gamma**2/2 * A) == '-\u03b3\xb2/2 A\u0302^(q\u2081)'
    #                                   -γ²/2 Â^(q₁)
    assert (unicode(tr(A * C, over_space=hs2)) ==
            'tr_(q\u2082)[C\u0302^(q\u2082)] A\u0302^(q\u2081)')
    #       tr_(q₂)[Ĉ^(q₂)] Â^(q₁)
    assert unicode(Adjoint(A)) == 'A\u0302^(q\u2081)\u2020'
    #                             Â^(q₁)†
    assert unicode(Adjoint(Create(hs=1))) == 'a\u0302\u207d\xb9\u207e'
    #              â⁽¹⁾
    assert unicode(PseudoInverse(A)) == '(A\u0302^(q\u2081))^+'
    #              (Â^(q₁))^+
    assert unicode(NullSpaceProjector(A)) == 'P\u0302_Ker(A\u0302^(q\u2081))'
    #                                         P̂_Ker(Â^(q₁))
    assert unicode(A - B) == 'A\u0302^(q\u2081) - B\u0302^(q\u2081)'
    #                         Â^(q₁) - B̂^(q₁)
    assert (unicode(2 * A - sqrt(gamma) * (B + C)) in [
            '2 A\u0302^(q\u2081) - \u221a\u03b3 (B\u0302^(q\u2081) '
            '+ C\u0302^(q\u2082))',
            '2 A\u0302^(q\u2081) - sqrt(\u03b3) (B\u0302^(q\u2081) '
            '+ C\u0302^(q\u2082))'])
    #       2 Â^(q₁) - √γ (B̂^(q₁) + Ĉ^(q₂))
    assert (unicode(Commutator(A, B)) ==
            '[A\u0302^(q\u2081), B\u0302^(q\u2081)]')
    #       [Â^(q₁), B̂^(q₁)]
    expr = (Commutator(A, B) * psi).dag()
    assert unicode(expr, show_hs_label=False) == r'⟨Ψ| [Â, B̂]^†'


def test_unicode_ket_elements():
    """Test the unicode representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    alpha, beta = symbols('alpha, beta')
    psi_hs1 = KetSymbol('Psi', hs=hs1)
    assert unicode(psi_hs1) == '|Ψ⟩^(q₁)'
    assert unicode(psi_hs1, unicode_sub_super=False) == '|Ψ⟩^(q_1)'
    assert unicode(KetSymbol('Psi', hs=1)) == '|Ψ⟩⁽¹⁾'
    assert unicode(KetSymbol('Psi', alpha, beta, hs=1)) == '|Ψ(α, β)⟩⁽¹⁾'
    assert unicode(KetSymbol('Psi', hs=(1, 2))) == '|Ψ⟩^(1⊗2)'
    assert unicode(KetSymbol('Psi', hs=hs1*hs2)) == '|Ψ⟩^(q₁⊗q₂)'
    assert unicode(ZeroKet) == '0'
    assert unicode(TrivialKet) == '1'
    assert unicode(BasisKet('e', hs=hs1)) == '|e⟩^(q₁)'
    assert unicode(BasisKet(1, hs=1)) == '|1⟩⁽¹⁾'
    assert unicode(CoherentStateKet(2, hs=1)) == '|α=2⟩⁽¹⁾'
    assert unicode(CoherentStateKet(2.0, hs=1)) == '|α=2⟩⁽¹⁾'
    unicode.printer.cache = {}
    assert unicode(CoherentStateKet(2.0, hs=1)) == '|α=2⟩⁽¹⁾'
    assert unicode(CoherentStateKet(2.1, hs=1)) == '|α=2.1⟩⁽¹⁾'


def test_unicode_symbolic_labels():
    """Test unicode representation of symbols with symbolic labels"""
    i = IdxSym('i')
    j = IdxSym('j')
    hs0 = LocalSpace(0)
    hs1 = LocalSpace(1)
    Psi = IndexedBase('Psi')
    assert unicode(BasisKet(FockIndex(2 * i), hs=hs0)) == '|2 i⟩⁽⁰⁾'
    assert unicode(KetSymbol(StrLabel(2 * i), hs=hs0)) == '|2 i⟩⁽⁰⁾'
    assert (
        unicode(KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1)) == '|Ψ_ij⟩^(0⊗1)')
    expr = BasisKet(FockIndex(i), hs=hs0) * BasisKet(FockIndex(j), hs=hs1)
    assert unicode(expr) == '|i,j⟩^(0⊗1)'
    assert unicode(Bra(BasisKet(FockIndex(2 * i), hs=hs0))) == '⟨2 i|⁽⁰⁾'
    assert (
        unicode(LocalSigma(FockIndex(i), FockIndex(j), hs=hs0)) == '|i⟩⟨j|⁽⁰⁾')
    expr = CoherentStateKet(symbols('alpha'), hs=1).to_fock_representation()
    assert unicode(expr) == 'exp(-α α ⃰/2) (∑_{n ∈ ℌ₁} αⁿ/√n! |n⟩⁽¹⁾)'

    tls = SpinSpace(label='s', spin='1/2', basis=('down', 'up'))
    Sig = IndexedBase('sigma')
    n = IdxSym('n')
    Sig_n = OperatorSymbol(StrLabel(Sig[n]), hs=tls)
    assert unicode(Sig_n, show_hs_label=False) == 'σ̂ₙ'


def test_unicode_bra_elements():
    """Test the unicode representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    alpha, beta = symbols('alpha, beta')
    assert unicode(Bra(KetSymbol('Psi', hs=hs1))) == '⟨Ψ|^(q₁)'
    assert unicode(Bra(KetSymbol('Psi', hs=1))) == '⟨Ψ|⁽¹⁾'
    assert unicode(Bra(KetSymbol('Psi', alpha, beta, hs=hs1))) == (
        '⟨Ψ(α, β)|^(q₁)')
    assert unicode(Bra(KetSymbol('Psi', hs=(1, 2)))) == '⟨Ψ|^(1⊗2)'
    assert unicode(Bra(ZeroKet)) == '0'
    assert unicode(Bra(TrivialKet)) == '1'
    assert unicode(BasisKet('e', hs=hs1).adjoint()) == '⟨e|^(q₁)'
    assert unicode(BasisKet(1, hs=1).adjoint()) == '⟨1|⁽¹⁾'
    assert unicode(CoherentStateKet(2.0, hs=1).dag()) == '⟨α=2|⁽¹⁾'
    assert unicode(CoherentStateKet(0.5j, hs=1).dag()) == '⟨α=0.5j|⁽¹⁾'
    assert unicode(CoherentStateKet(I/2, hs=1).dag()) == '⟨α=ⅈ/2|⁽¹⁾'


def test_unicode_ket_operations():
    """Test the unicode representation of ket operations"""
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    alpha = symbols('alpha')
    beta = symbols('beta')
    phase = exp(-I * gamma)
    i = IdxSym('i')
    assert unicode(psi1 + psi2) == '|Ψ₁⟩^(q₁) + |Ψ₂⟩^(q₁)'
    assert unicode(psi1 * phi) == '|Ψ₁⟩^(q₁) ⊗ |Φ⟩^(q₂)'
    assert unicode(phase * psi1) == 'exp(-ⅈ γ) |Ψ₁⟩^(q₁)'
    assert (
        unicode((alpha + 1) * KetSymbol('Psi', hs=0)) == '(α + 1) |Ψ⟩⁽⁰⁾')
    assert (unicode(A * psi1) ==
            'A\u0302_0^(q\u2081) |\u03a8\u2081\u27e9^(q\u2081)')
    #        Â_0^(q₁) |Ψ₁⟩^(q₁)
    assert unicode(BraKet(psi1, psi2)) == '⟨Ψ₁|Ψ₂⟩^(q₁)'
    expr = BraKet(
        KetSymbol('Psi_1', alpha, hs=hs1), KetSymbol('Psi_2', beta, hs=hs1))
    assert unicode(expr) == '⟨Ψ₁(α)|Ψ₂(β)⟩^(q₁)'
    assert unicode(ket_e1.dag() * ket_e1) == '1'
    assert unicode(ket_g1.dag() * ket_e1) == '0'
    assert unicode(KetBra(psi1, psi2)) == '|Ψ₁⟩⟨Ψ₂|^(q₁)'
    expr = KetBra(
        KetSymbol('Psi_1', alpha, hs=hs1), KetSymbol('Psi_2', beta, hs=hs1))
    assert unicode(expr) == '|Ψ₁(α)⟩⟨Ψ₂(β)|^(q₁)'
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    assert (unicode(bell1) ==
            '1/√2 (|eg⟩^(q₁⊗q₂) - ⅈ |ge⟩^(q₁⊗q₂))')
    assert (unicode(BraKet.create(bell1, bell2)) ==
            r'1/2 (⟨eg|^(q₁⊗q₂) + ⅈ ⟨ge|^(q₁⊗q₂)) (|ee⟩^(q₁⊗q₂) - '
            r'|gg⟩^(q₁⊗q₂))')
    assert (unicode(KetBra.create(bell1, bell2)) ==
            r'1/2 (|eg⟩^(q₁⊗q₂) - ⅈ |ge⟩^(q₁⊗q₂))(⟨ee|^(q₁⊗q₂) - '
            r'⟨gg|^(q₁⊗q₂))')
    assert (unicode(KetBra.create(bell1, bell2), show_hs_label=False) ==
            r'1/2 (|eg⟩ - ⅈ |ge⟩)(⟨ee| - ⟨gg|)')
    expr = KetBra(KetSymbol('Psi', hs=0), BasisKet(FockIndex(i), hs=0))
    assert unicode(expr) == "|Ψ⟩⟨i|⁽⁰⁾"
    expr = KetBra(BasisKet(FockIndex(i), hs=0), KetSymbol('Psi', hs=0))
    assert unicode(expr) == "|i⟩⟨Ψ|⁽⁰⁾"
    expr = BraKet(KetSymbol('Psi', hs=0), BasisKet(FockIndex(i), hs=0))
    assert unicode(expr) == "⟨Ψ|i⟩⁽⁰⁾"
    expr = BraKet(BasisKet(FockIndex(i), hs=0), KetSymbol('Psi', hs=0))
    assert unicode(expr) == "⟨i|Ψ⟩⁽⁰⁾"


def test_unicode_bra_operations():
    """Test the unicode representation of bra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert unicode((psi1 + psi2).dag()) == '⟨Ψ₁|^(q₁) + ⟨Ψ₂|^(q₁)'
    assert unicode((psi1 * phi).dag()) == '⟨Ψ₁|^(q₁) ⊗ ⟨Φ|^(q₂)'
    assert unicode(Bra(phase * psi1)) == 'exp(ⅈ γ) ⟨Ψ₁|^(q₁)'


def test_unicode_sop_elements():
    """Test the unicode representation of "atomic" Superoperators"""
    hs1 = LocalSpace('q1', dimension=2)
    alpha, beta = symbols('alpha, beta')
    assert unicode(SuperOperatorSymbol("A", hs=hs1)) == 'A^(q₁)'
    assert (unicode(SuperOperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            'Ξ_2^(q₁⊗q₂)')
    assert (unicode(SuperOperatorSymbol("Xi", alpha, beta, hs=hs1)) ==
            'Ξ^(q₁)(α, β)')
    assert unicode(IdentitySuperOperator) == "𝟙"
    assert unicode(ZeroSuperOperator) == "0"


def test_unicode_sop_operations():
    """Test the unicode representation of super operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = SuperOperatorSymbol("A", hs=hs1)
    B = SuperOperatorSymbol("B", hs=hs1)
    C = SuperOperatorSymbol("C", hs=hs2)
    L = SuperOperatorSymbol("L", hs=1)
    M = SuperOperatorSymbol("M", hs=1)
    A_op = OperatorSymbol("A", hs=1)
    gamma = symbols('gamma', positive=True)
    assert unicode(A + B) == 'A^(q₁) + B^(q₁)'
    assert unicode(A * B) == 'A^(q₁) B^(q₁)'
    assert unicode(A * C) == 'A^(q₁) C^(q₂)'
    assert unicode(2j * A) == '2j A^(q₁)'
    assert unicode(gamma**2 * A) == 'γ² A^(q₁)'
    assert unicode(SuperAdjoint(A)) == 'A^(q₁)†'
    assert unicode(A - B + C) == 'A^(q₁) - B^(q₁) + C^(q₂)'
    assert (unicode(2 * A - sqrt(gamma) * (B + C)) in
            ['2 A^(q₁) - sqrt(γ) (B^(q₁) + C^(q₂))',
             '2 A^(q₁) - √γ (B^(q₁) + C^(q₂))'])
    assert unicode(SPre(A_op)) == 'SPre(A\u0302\u207d\xb9\u207e)'
    #                              SPre(Â⁽¹⁾)
    assert unicode(SPost(A_op)) == 'SPost(A\u0302\u207d\xb9\u207e)'
    #                               SPost(Â⁽¹⁾)
    assert (unicode(SuperOperatorTimesOperator(L, A_op)) ==
            'L\u207d\xb9\u207e[A\u0302\u207d\xb9\u207e]')
    #        L⁽¹⁾[Â⁽¹⁾]
    assert (unicode(SuperOperatorTimesOperator(L, sqrt(gamma) * A_op)) in
            ['L\u207d\xb9\u207e[\u221a\u03b3 A\u0302\u207d\xb9\u207e]',
             'L\u207d\xb9\u207e[sqrt(\u03b3) A\u0302\u207d\xb9\u207e]'])
     #        L⁽¹⁾[√γ Â⁽¹⁾]
    assert (unicode(SuperOperatorTimesOperator((L + 2*M), A_op)) ==
            '(L\u207d\xb9\u207e + 2 M\u207d\xb9\u207e)'
            '[A\u0302\u207d\xb9\u207e]')
    #       (L⁽¹⁾ + 2 M⁽¹⁾)[Â⁽¹⁾]


def test_unicode_spin_arrows():
    """Test the representation of spin-1/2 spaces with special labels "down",
    "up" as arrows"""
    tls1 = SpinSpace('1', spin='1/2', basis=("down", "up"))
    tls2 = SpinSpace('2', spin='1/2', basis=("down", "up"))
    tls3 = SpinSpace('3', spin='1/2', basis=("down", "up"))
    down1 = BasisKet('down', hs=tls1)
    up1 = BasisKet('up', hs=tls1)
    down2 = BasisKet('down', hs=tls2)
    up3 = BasisKet('up', hs=tls3)
    assert unicode(down1) == r'|↓⟩⁽¹⁾'
    assert unicode(up1) == r'|↑⟩⁽¹⁾'
    ket = down1 * down2 * up3
    assert unicode(ket) == r'|↓↓↑⟩^(1⊗2⊗3)'
    sig = LocalSigma("up", "down", hs=tls1)
    assert unicode(sig) == r'|↑⟩⟨↓|⁽¹⁾'


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

        def _unicode(self, *args, **kwargs):
            return "%s(%s)" % (
                self._name, ", ".join(
                    [unicode(sym) for sym in self._sym_args]))

    return ScalarFunc


def test_unicode_derivative(MyScalarFunc):
    s, s0, t, t0, gamma = symbols('s, s_0, t, t_0, gamma', real=True)
    m = IdxSym('m')
    n = IdxSym('n')
    S = IndexedBase('s')
    T = IndexedBase('t')

    f = partial(MyScalarFunc, "f")
    g = partial(MyScalarFunc, "g")

    expr = f(s, t).diff(s, n=2).diff(t)
    assert unicode(expr) == '∂ₛ² ∂ₜ f(s, t)'

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: s0})
    assert unicode(expr) == '∂ₛ² ∂ₜ f(s, t) |_s=s₀'

    expr = f(S[m], T[n]).diff(S[m], n=2).diff(T[n]).evaluate_at({S[m]: s0})
    assert unicode(expr) == '∂_sₘ^2 ∂_tₙ f(sₘ, tₙ) |_sₘ=s₀'

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: 0})
    assert unicode(expr) == '∂ₛ² ∂ₜ f(s, t) |ₛ₌₀'

    expr = f(gamma, t).diff(gamma, n=2).diff(t).evaluate_at({gamma: 0})
    assert unicode(expr) == '∂ᵧ² ∂ₜ f(γ, t) |ᵧ₌₀'

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: s0, t: t0})
    assert unicode(expr) == '∂ₛ² ∂ₜ f(s, t) |_(s=s₀, t=t₀)'

    D = expr.__class__

    expr = D(f(s, t) + g(s, t), derivs={s: 2, t: 1}, vals={s: s0, t: t0})
    assert unicode(expr) == '∂ₛ² ∂ₜ (f(s, t) + g(s, t)) |_(s=s₀, t=t₀)'

    expr = D(2 * f(s, t), derivs={s: 2, t: 1}, vals={s: s0, t: t0})
    assert unicode(expr) == '∂ₛ² ∂ₜ (2 f(s, t)) |_(s=s₀, t=t₀)'

    expr = f(s, t).diff(t) + g(s, t)
    assert unicode(expr) == '∂ₜ f(s, t) + g(s, t)'

    expr = f(s, t).diff(t) * g(s, t)
    assert unicode(expr) == '(∂ₜ f(s, t)) g(s, t)'

    expr = f(s, t).diff(t).evaluate_at({t: 0}) * g(s, t)
    assert unicode(expr) == '(∂ₜ f(s, t) |ₜ₌₀) g(s, t)'

    f = MyScalarFunc("f", S[m], T[n])
    series = f.series_expand(T[n], about=0, order=3)
    assert unicode(series) == (
        '('
        'f(sₘ, 0), '
        '∂_tₙ f(sₘ, tₙ) |_tₙ=0, '
        '1/2 (∂_tₙ^2 f(sₘ, tₙ) |_tₙ=0), '
        '1/6 (∂_tₙ^3 f(sₘ, tₙ) |_tₙ=0))')

    f = MyScalarFunc("f", s, t)
    series = f.series_expand(t, about=0, order=2)
    assert (
        unicode(series) ==
        '(f(s, 0), ∂ₜ f(s, t) |ₜ₌₀, 1/2 (∂ₜ² f(s, t) |ₜ₌₀))')

    f = MyScalarFunc("f", S[m], T[n])
    series = f.series_expand(T[n], about=t0, order=3)
    assert unicode(series) == (
        '('
        'f(sₘ, t₀), '
        '∂_tₙ f(sₘ, tₙ) |_tₙ=t₀, '
        '1/2 (∂_tₙ^2 f(sₘ, tₙ) |_tₙ=t₀), '
        '1/6 (∂_tₙ^3 f(sₘ, tₙ) |_tₙ=t₀))')

    expr = (  # nested derivative
        MyScalarFunc("f", s, t)
        .diff(s, n=2)
        .diff(t)
        .evaluate_at({t: t0})
        .diff(t0))
    assert unicode(expr) == '∂_t₀ (∂ₛ² ∂ₜ f(s, t) |_t=t₀)'
