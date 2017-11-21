# This file is part of QNET.
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

import sympy
from sympy import symbols, sqrt, exp, I, Rational

from qnet.algebra.circuit_algebra import(
        CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
        Feedback, SeriesInverse, cid)
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.three_port_kerr_cavity_cc import (
        ThreePortKerrCavity)
from qnet.algebra.operator_algebra import(
        OperatorSymbol, IdentityOperator, ZeroOperator, Create, Destroy, Jz,
        Jplus, Jminus, Phase, Displace, Squeeze, LocalSigma, tr, Adjoint,
        PseudoInverse, NullSpaceProjector, Commutator)
from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, TrivialSpace, FullSpace)
from qnet.algebra.matrix_algebra import Matrix
from qnet.algebra.state_algebra import (
        KetSymbol, LocalKet, ZeroKet, TrivialKet, BasisKet, CoherentStateKet,
        UnequalSpaces, ScalarTimesKet, OperatorTimesKet, Bra,
        OverlappingSpaces, SpaceTooLargeError, BraKet, KetBra)
from qnet.algebra.super_operator_algebra import (
        SuperOperatorSymbol, IdentitySuperOperator, ZeroSuperOperator,
        SuperAdjoint, SPre, SPost, SuperOperatorTimesOperator)
from qnet.printing import unicode


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


def test_unicode_circuit_elements():
    """Test the unicode representation of "atomic" circuit algebra elements"""
    assert unicode(CircuitSymbol("C", cdim=2)) == 'C'
    C_1 = CircuitSymbol("C_1", cdim=2)
    assert unicode(C_1) == 'C₁'
    assert unicode(C_1, unicode_sub_super=False) == 'C_1'
    assert unicode(CircuitSymbol("Xi_2", 2)) == 'Ξ₂'
    assert unicode(CircuitSymbol("Xi_full", 2)) == 'Ξ_full'
    assert unicode(CIdentity) == 'CIdentity'
    assert unicode(cid(4)) == 'cid(4)'
    assert unicode(CircuitZero) == 'CircuitZero'


def test_unicode_circuit_components():
    """Test ascii-printing of some of the circuit components"""
    B11 = Beamsplitter('Latch.B11')
    assert unicode(B11) == 'Latch.B11(theta=π/4)'
    C1 = ThreePortKerrCavity('Latch.C1')
    assert (
        unicode(C1) == 'Latch.C1(Delta=Δ, chi=χ, kappa_1=κ₁, kappa_2=κ₂, '
        'kappa_3=κ₃, FOCK_DIM=75)')
    A = CircuitSymbol("A", cdim=2)
    expr = A << B11
    assert unicode(expr) == 'A ◁ Latch.B11(theta=π/4)'
    expr = Feedback(Beamsplitter('BS'), out_port=1, in_port=0)
    assert unicode(expr) == '[BS(theta=π/4)]₁₋₀'


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


def test_unicode_operator_elements():
    """Test the unicode representation of "atomic" operator algebra elements"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    assert unicode(OperatorSymbol("A", hs=hs1)) == 'A\u0302^(q\u2081)'
    #                                               Â^(q₁)
    assert unicode(
        OperatorSymbol("A", hs=hs1),
        operator_hats=False, unicode_sub_super=False) == 'A^(q_1)'
    assert (unicode(OperatorSymbol("A_1", hs=hs1*hs2)) ==
            'A\u0302_1^(q\u2081\u2297q\u2082)')  # Â_1^(q₁⊗q₂)
    assert (unicode(OperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            '\u039e\u0302_2^(q\u2081\u2297q\u2082)')  # Ξ̂_2^(q₁⊗q₂)
    assert unicode(IdentityOperator) == "𝟙"
    assert unicode(ZeroOperator) == "0"
    assert unicode(Create(hs=1)) == 'a\u0302^(1)\u2020'  # â^(1)†
    assert unicode(Destroy(hs=1)) == 'a\u0302\u207d\xb9\u207e'  # â⁽¹⁾
    assert (
        unicode(Destroy(hs=1), unicode_sub_super=False) == 'a\u0302^(1)')
    assert (
        unicode(Destroy(hs=1), operator_hats=False) == 'a\u207d\xb9\u207e')
    assert (
        unicode(
            Destroy(hs=1), operator_hats=False, unicode_sub_super=False) ==
        'a^(1)')
    assert (unicode(Squeeze(Rational(1, 2), hs=1)) ==
            'Squeeze\u207d\xb9\u207e(1/2)')
    #       Squeeze⁽¹⁾(1/2)
    hs_tls = LocalSpace('1', basis=('g', 'e'))
    sig_e_g = LocalSigma('e', 'g', hs=hs_tls)
    assert unicode(sig_e_g) == '|e⟩⟨g|⁽¹⁾'
    assert unicode(sig_e_g, unicode_sub_super=False) == '|e⟩⟨g|^(1)'
    assert unicode(sig_e_g, show_hilbert_space=False) == '|e⟩⟨g|'
    assert (
        unicode(sig_e_g, local_sigma_as_ketbra=False) ==
        '\u03c3\u0302_e,g^(1)')  # σ̂_e,g^(1)
    sig_e_e = LocalSigma('e', 'e', hs=hs_tls)
    assert unicode(sig_e_e) == '|e⟩⟨e|⁽¹⁾'
    assert (
        unicode(sig_e_e, local_sigma_as_ketbra=False) ==
        '\u03a0\u0302\u2091\u207d\xb9\u207e')  # Π̂ₑ⁽¹⁾


def test_unicode_operator_operations():
    """Test the unicode representation of operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    gamma = symbols('gamma', positive=True)
    assert unicode(A + B) == 'A\u0302^(q\u2081) + B\u0302^(q\u2081)'
    #                         Â^(q₁) + B̂^(q₁)
    assert unicode(A * B) == 'A\u0302^(q\u2081) * B\u0302^(q\u2081)'
    #                         Â^(q₁) B̂^(q₁)
    assert unicode(A * C) == 'A\u0302^(q\u2081) * C\u0302^(q\u2082)'
    #                         Â^(q₁) * Ĉ^(q₂)
    assert unicode(2 * A) == '2 * A\u0302^(q\u2081)'  # 2 * Â^(q₁)
    assert unicode(2j * A) == '2j * A\u0302^(q\u2081)'
    #                          2j * Â^(q₁)
    assert unicode((1+2j) * A) == '(1+2j) * A\u0302^(q\u2081)'
    #                              (1+2j) * Â^(q₁)
    assert unicode(gamma**2 * A) == '\u03b3\xb2 * A\u0302^(q\u2081)'
    #                                γ² * Â^(q₁)
    assert unicode(-gamma**2/2 * A) == '-\u03b3\xb2/2 * A\u0302^(q\u2081)'
    #                                   -γ²/2 * Â^(q₁)
    assert (unicode(tr(A * C, over_space=hs2)) ==
            'tr_(q\u2082)[C\u0302^(q\u2082)] * A\u0302^(q\u2081)')
    #       tr_(q₂)[Ĉ^(q₂)] * Â^(q₁)
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
            '2 * A\u0302^(q\u2081) - \u221a\u03b3 * (B\u0302^(q\u2081) '
            '+ C\u0302^(q\u2082))',
            '2 * A\u0302^(q\u2081) - sqrt(\u03b3) * (B\u0302^(q\u2081) '
            '+ C\u0302^(q\u2082))'])
    #       2 * Â^(q₁) - √γ * (B̂^(q₁) + Ĉ^(q₂))
    assert (unicode(Commutator(A, B)) ==
            '[A\u0302^(q\u2081), B\u0302^(q\u2081)]')
    #       [Â^(q₁), B̂^(q₁)]


def test_unicode_ket_elements():
    """Test the unicode representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    psi_hs1 = KetSymbol('Psi', hs=hs1)
    assert unicode(psi_hs1) == '|Ψ⟩^(q₁)'
    assert unicode(psi_hs1, unicode_sub_super=False) == '|Ψ⟩^(q_1)'
    assert unicode(KetSymbol('Psi', hs=1)) == '|Ψ⟩⁽¹⁾'
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


def test_unicode_bra_elements():
    """Test the unicode representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    assert unicode(Bra(KetSymbol('Psi', hs=hs1))) == '⟨Ψ|^(q₁)'
    assert unicode(Bra(KetSymbol('Psi', hs=1))) == '⟨Ψ|⁽¹⁾'
    assert unicode(Bra(KetSymbol('Psi', hs=(1, 2)))) == '⟨Ψ|^(1⊗2)'
    assert unicode(Bra(ZeroKet)) == '0'
    assert unicode(Bra(TrivialKet)) == '1'
    assert unicode(BasisKet('e', hs=hs1).adjoint()) == '⟨e|^(q₁)'
    assert unicode(BasisKet(1, hs=1).adjoint()) == '⟨1|⁽¹⁾'
    assert unicode(CoherentStateKet(2.0, hs=1).dag) == '⟨α=2|⁽¹⁾'
    assert unicode(CoherentStateKet(0.5j, hs=1).dag) == '⟨α=0.5j|⁽¹⁾'
    assert unicode(CoherentStateKet(I/2, hs=1).dag) == '⟨α=ⅈ/2|⁽¹⁾'


def test_unicode_ket_operations():
    """Test the unicode representation of ket operations"""
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi1_l = LocalKet("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    phi_l = LocalKet("Phi", hs=hs2)
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert unicode(psi1 + psi2) == '|Ψ₁⟩^(q₁) + |Ψ₂⟩^(q₁)'
    assert unicode(psi1 * phi) == '|Ψ₁⟩^(q₁) ⊗ |Φ⟩^(q₂)'
    assert unicode(psi1_l * phi_l) == '|Ψ₁,Φ⟩^(q₁⊗q₂)'
    assert unicode(phase * psi1) == 'exp(-ⅈ γ) * |Ψ₁⟩^(q₁)'
    assert (unicode(A * psi1) ==
            'A\u0302_0^(q\u2081) |\u03a8\u2081\u27e9^(q\u2081)')
    #        Â_0^(q₁) |Ψ₁⟩^(q₁)
    assert unicode(BraKet(psi1, psi2)) == '⟨Ψ₁|Ψ₂⟩^(q₁)'
    assert unicode(ket_e1.dag * ket_e1) == '1'
    assert unicode(ket_g1.dag * ket_e1) == '0'
    assert unicode(KetBra(psi1, psi2)) == '|Ψ₁⟩⟨Ψ₂|^(q₁)'
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    assert (unicode(bell1) ==
            '1/√2 * (|e,g⟩^(q₁⊗q₂) - ⅈ * |g,e⟩^(q₁⊗q₂))')
    assert (unicode(BraKet.create(bell1, bell2)) ==
            r'1/2 * (⟨e,g|^(q₁⊗q₂) + ⅈ * ⟨g,e|^(q₁⊗q₂)) * (|e,e⟩^(q₁⊗q₂) - '
            r'|g,g⟩^(q₁⊗q₂))')
    assert (unicode(KetBra.create(bell1, bell2)) ==
            r'1/2 * (|e,g⟩^(q₁⊗q₂) - ⅈ * |g,e⟩^(q₁⊗q₂))(⟨e,e|^(q₁⊗q₂) - '
            r'⟨g,g|^(q₁⊗q₂))')
    assert (unicode(KetBra.create(bell1, bell2), show_hilbert_space=False) ==
            r'1/2 * (|e,g⟩ - ⅈ * |g,e⟩)(⟨e,e| - ⟨g,g|)')


def test_unicode_bra_operations():
    """Test the unicode representation of bra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    bra_psi1_l = LocalKet("Psi_1", hs=hs1).dag
    bra_phi_l = LocalKet("Phi", hs=hs2).dag
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert unicode((psi1 + psi2).dag) == '⟨Ψ₁|^(q₁) + ⟨Ψ₂|^(q₁)'
    assert unicode((psi1 * phi).dag) == '⟨Ψ₁|^(q₁) ⊗ ⟨Φ|^(q₂)'
    assert unicode(bra_psi1_l * bra_phi_l) == '⟨Ψ₁,Φ|^(q₁⊗q₂)'
    assert unicode(Bra(phase * psi1)) == 'exp(ⅈ γ) * ⟨Ψ₁|^(q₁)'


def test_unicode_sop_elements():
    """Test the unicode representation of "atomic" Superoperators"""
    hs1 = LocalSpace('q1', dimension=2)
    assert unicode(SuperOperatorSymbol("A", hs=hs1)) == 'A^(q₁)'
    assert (unicode(SuperOperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            'Ξ_2^(q₁⊗q₂)')
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
    assert unicode(A * B) == 'A^(q₁) * B^(q₁)'
    assert unicode(A * C) == 'A^(q₁) * C^(q₂)'
    assert unicode(2j * A) == '2j * A^(q₁)'
    assert unicode(gamma**2 * A) == 'γ² * A^(q₁)'
    assert unicode(SuperAdjoint(A)) == 'A^(q₁)†'
    assert unicode(A - B + C) == 'A^(q₁) - B^(q₁) + C^(q₂)'
    assert (unicode(2 * A - sqrt(gamma) * (B + C)) in
            ['2 * A^(q₁) - sqrt(γ) * (B^(q₁) + C^(q₂))',
             '2 * A^(q₁) - √γ * (B^(q₁) + C^(q₂))'])
    assert unicode(SPre(A_op)) == 'SPre(A\u0302\u207d\xb9\u207e)'
    #                              SPre(Â⁽¹⁾)
    assert unicode(SPost(A_op)) == 'SPost(A\u0302\u207d\xb9\u207e)'
    #                               SPost(Â⁽¹⁾)
    assert (unicode(SuperOperatorTimesOperator(L, A_op)) ==
            'L\u207d\xb9\u207e[A\u0302\u207d\xb9\u207e]')
    #        L⁽¹⁾[Â⁽¹⁾]
    assert (unicode(SuperOperatorTimesOperator(L, sqrt(gamma) * A_op)) in
            ['L\u207d\xb9\u207e[\u221a\u03b3 * A\u0302\u207d\xb9\u207e]',
             'L\u207d\xb9\u207e[sqrt(\u03b3) * A\u0302\u207d\xb9\u207e]'])
     #        L⁽¹⁾[√γ * Â⁽¹⁾]
    assert (unicode(SuperOperatorTimesOperator((L + 2*M), A_op)) ==
            '(L\u207d\xb9\u207e + 2 * M\u207d\xb9\u207e)'
            '[A\u0302\u207d\xb9\u207e]')
    #       (L⁽¹⁾ + 2 * M⁽¹⁾)[Â⁽¹⁾]
