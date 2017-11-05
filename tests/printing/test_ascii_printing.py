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

import pytest

from sympy import symbols, sqrt, exp, I

from qnet.algebra.circuit_algebra import(
        CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
        Feedback, SeriesInverse)
from qnet.algebra.operator_algebra import(
        OperatorSymbol, IdentityOperator, ZeroOperator, Create, Destroy, Jz,
        Jplus, Jminus, Phase, Displace, Squeeze, LocalSigma, LocalProjector,
        tr, Adjoint, PseudoInverse, NullSpaceProjector, Commutator)
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
from qnet.printing import ascii


def test_ascii_circuit_elements():
    """Test the ascii representation of "atomic" circuit algebra elements"""
    assert ascii(CircuitSymbol("C", cdim=2)) == 'C'
    assert ascii(CircuitSymbol("C_1", cdim=2)) == 'C_1'
    assert ascii(CircuitSymbol("Xi_2", 2)) == 'Xi_2'
    assert ascii(CircuitSymbol("Xi_full", 2)) == 'Xi_full'
    with pytest.raises(ValueError):
        CircuitSymbol(r'\Xi^2', 2)
    assert ascii(CIdentity) == 'cid(1)'
    assert ascii(CircuitZero) == 'cid(0)'


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
    assert ascii(A + (B << C)) == "A_test + B_test << C_test"
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
        LocalSpace('\kappa')
    assert ascii(TrivialSpace) == 'H_null'
    assert ascii(FullSpace) == 'H_total'


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


def test_ascii_operator_elements():
    """Test the ascii representation of "atomic" operator algebra elements"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    assert ascii(OperatorSymbol("A", hs=hs1)) == 'A^(q1)'
    A_1 = OperatorSymbol("A_1", hs=1)
    assert ascii(A_1, show_hilbert_space='subscript') == 'A_1,(1)'
    assert ascii(OperatorSymbol("A", hs=hs1), show_hilbert_space=False) == 'A'
    assert ascii(OperatorSymbol("A_1", hs=hs1*hs2)) == 'A_1^(q1*q2)'
    assert ascii(OperatorSymbol("Xi_2", hs=('q1', 'q2'))) == 'Xi_2^(q1*q2)'
    assert ascii(OperatorSymbol("Xi_full", hs=1)) == 'Xi_full^(1)'
    with pytest.raises(ValueError):
        OperatorSymbol(r'\Xi^2', hs='a')
    assert ascii(IdentityOperator) == "1"
    assert ascii(ZeroOperator) == "0"
    assert ascii(Create(hs=1)) == "a^(1)H"
    assert ascii(Create(hs=1), show_hilbert_space=False) == "a^H"
    assert ascii(Create(hs=1), show_hilbert_space='subscript') == "a_(1)^H"
    assert ascii(Create(hs=1, identifier='b')) == "b^(1)H"
    assert ascii(Destroy(hs=1)) == "a^(1)"
    assert ascii(Destroy(hs=1, identifier='b')) == "b^(1)"
    assert ascii(Jz(hs=1)) == "J_z^(1)"
    assert ascii(Jz(hs=1, identifier='Z')) == "Z^(1)"
    assert ascii(Jplus(hs=1, identifier='Jp')) == "Jp^(1)"
    assert ascii(Jminus(hs=1, identifier='Jm')) == "Jm^(1)"
    assert ascii(Phase(0.5, hs=1)) == 'Phase^(1)(0.5)'
    assert ascii(Phase(0.5, hs=1, identifier='Ph')) == 'Ph^(1)(0.5)'
    assert ascii(Displace(0.5, hs=1)) == 'D^(1)(0.5)'
    assert ascii(Squeeze(0.5, hs=1)) == 'Squeeze^(1)(0.5)'
    hs_tls = LocalSpace('1', basis=('g', 'e'))
    sig_e_g = LocalSigma('e', 'g', hs=hs_tls)
    assert ascii(sig_e_g) == '|e><g|^(1)'
    assert ascii(sig_e_g, local_sigma_as_ketbra=False) == 'sigma_e,g^(1)'
    sig_e_e = LocalProjector('e', hs=hs_tls)
    assert ascii(sig_e_e, local_sigma_as_ketbra=False) == 'Pi_e^(1)'


def test_ascii_operator_operations():
    """Test the ascii representation of operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    gamma = symbols('gamma', positive=True)
    assert ascii(A + B) == 'A^(q_1) + B^(q_1)'
    assert ascii(A * B) == 'A^(q_1) * B^(q_1)'
    assert ascii(A * C) == 'A^(q_1) * C^(q_2)'
    assert ascii(2 * A) == '2 * A^(q_1)'
    assert ascii(2j * A) == '2j * A^(q_1)'
    assert ascii((1+2j) * A) == '(1+2j) * A^(q_1)'
    assert ascii(gamma**2 * A) == 'gamma**2 * A^(q_1)'
    assert ascii(-gamma**2/2 * A) == '-gamma**2/2 * A^(q_1)'
    assert ascii(tr(A * C, over_space=hs2)) == 'tr_(q_2)[C^(q_2)] * A^(q_1)'
    assert ascii(Adjoint(A)) == 'A^(q_1)H'
    assert ascii(Adjoint(Create(hs=1))) == 'a^(1)'
    assert ascii(Adjoint(A + B)) == '(A^(q_1) + B^(q_1))^H'
    assert ascii(PseudoInverse(A)) == '(A^(q_1))^+'
    assert ascii(NullSpaceProjector(A)) == 'P_Ker(A^(q_1))'
    assert ascii(A - B) == 'A^(q_1) - B^(q_1)'
    assert ascii(A - B + C) == 'A^(q_1) - B^(q_1) + C^(q_2)'
    assert (ascii(2 * A - sqrt(gamma) * (B + C)) ==
            '2 * A^(q_1) - sqrt(gamma) * (B^(q_1) + C^(q_2))')
    assert ascii(Commutator(A, B)) == r'[A^(q_1), B^(q_1)]'


def test_ascii_ket_elements():
    """Test the ascii representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    assert ascii(KetSymbol('Psi', hs=hs1)) == '|Psi>_(q1)'
    assert ascii(KetSymbol('Psi', hs=1)) == '|Psi>_(1)'
    assert ascii(KetSymbol('Psi', hs=(1, 2))) == '|Psi>_(1*2)'
    assert ascii(KetSymbol('Psi', hs=hs1*hs2)) == '|Psi>_(q1*q2)'
    with pytest.raises(ValueError):
        KetSymbol(r'\Psi', hs=hs1)
    assert ascii(LocalKet('Psi', hs=1)) == '|Psi>_(1)'
    with pytest.raises(ValueError):
        LocalKet('Psi', hs=hs1*hs2)
    assert ascii(ZeroKet) == '0'
    assert ascii(TrivialKet) == '1'
    assert ascii(BasisKet('e', hs=hs1)) == '|e>_(q1)'
    assert ascii(BasisKet(1, hs=1)) == '|1>_(1)'
    assert ascii(BasisKet(1, hs=hs1)) == '|e>_(q1)'
    with pytest.raises(ValueError):
        BasisKet('1', hs=hs1)
    assert ascii(CoherentStateKet(2.0, hs=1)) == '|alpha=2.0>_(1)'


def test_ascii_bra_elements():
    """Test the ascii representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    assert ascii(Bra(KetSymbol('Psi', hs=hs1))) == '<Psi|_(q1)'
    assert ascii(Bra(KetSymbol('Psi', hs=1))) == '<Psi|_(1)'
    assert ascii(Bra(KetSymbol('Psi', hs=(1, 2)))) == '<Psi|_(1*2)'
    assert ascii(Bra(KetSymbol('Psi', hs=hs1*hs2))) == '<Psi|_(q1*q2)'
    assert ascii(LocalKet('Psi', hs=1).dag) == '<Psi|_(1)'
    assert ascii(Bra(ZeroKet)) == '0'
    assert ascii(Bra(TrivialKet)) == '1'
    assert ascii(BasisKet('e', hs=hs1).adjoint()) == '<e|_(q1)'
    assert ascii(BasisKet(1, hs=1).adjoint()) == '<1|_(1)'
    assert ascii(CoherentStateKet(2.0, hs=1).dag) == '<alpha=2.0|_(1)'


def test_ascii_ket_operations():
    """Test the ascii representation of ket operations"""
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi1_l = LocalKet("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi3 = KetSymbol("Psi_3", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    phi_l = LocalKet("Phi", hs=hs2)
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert ascii(psi1 + psi2) == '|Psi_1>_(q_1) + |Psi_2>_(q_1)'
    assert (ascii(psi1 - psi2 + psi3) ==
            '|Psi_1>_(q_1) - |Psi_2>_(q_1) + |Psi_3>_(q_1)')
    with pytest.raises(UnequalSpaces):
        psi1 + phi
    with pytest.raises(AttributeError):
        (psi1 * phi).label
    assert ascii(psi1 * phi) == '|Psi_1>_(q_1) * |Phi>_(q_2)'
    assert (psi1_l * phi_l).label == 'Psi_1,Phi'
    assert ascii(psi1_l * phi_l) == '|Psi_1,Phi>_(q_1*q_2)'
    with pytest.raises(OverlappingSpaces):
        psi1 * psi2
    assert ascii(phase * psi1) == 'exp(-I*gamma) * |Psi_1>_(q_1)'
    assert ascii(A * psi1) == 'A_0^(q_1) |Psi_1>_(q_1)'
    with pytest.raises(SpaceTooLargeError):
        A * phi
    assert ascii(BraKet(psi1, psi2)) == '<Psi_1|Psi_2>_(q_1)'
    assert ascii(psi1.dag * psi2) == '<Psi_1|Psi_2>_(q_1)'
    assert ascii(ket_e1.dag * ket_e1) == '1'
    assert ascii(ket_g1.dag * ket_e1) == '0'
    assert ascii(KetBra(psi1, psi2)) == '|Psi_1><Psi_2|_(q_1)'
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    assert (ascii(bell1) ==
            'sqrt(2)/2 * (|e,g>_(q_1*q_2) - I * |g,e>_(q_1*q_2))')
    assert (ascii(bell2) ==
            'sqrt(2)/2 * (|e,e>_(q_1*q_2) - |g,g>_(q_1*q_2))')
    assert (ascii(BraKet.create(bell1, bell2)) ==
            r'1/2 * (<e,g|_(q_1*q_2) + I * <g,e|_(q_1*q_2))*(|e,e>_(q_1*q_2) '
            r'- |g,g>_(q_1*q_2))')
    assert (ascii(KetBra.create(bell1, bell2)) ==
            '1/2 * (|e,g>_(q_1*q_2) - I * |g,e>_(q_1*q_2))(<e,e|_(q_1*q_2) '
            '- <g,g|_(q_1*q_2))')


def test_ascii_bra_operations():
    """Test the ascii representation of bra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi3 = KetSymbol("Psi_3", hs=hs1)
    phi = KetSymbol("Phi", hs=hs2)
    bra_psi1 = KetSymbol("Psi_1", hs=hs1).dag
    bra_psi1_l = LocalKet("Psi_1", hs=hs1).dag
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag
    bra_psi3 = KetSymbol("Psi_3", hs=hs1).dag
    bra_phi = KetSymbol("Phi", hs=hs2).dag
    bra_phi_l = LocalKet("Phi", hs=hs2).dag
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert ascii((psi1 + psi2).dag) == '<Psi_1|_(q_1) + <Psi_2|_(q_1)'
    assert ascii(bra_psi1 + bra_psi2) == '<Psi_1|_(q_1) + <Psi_2|_(q_1)'
    assert (ascii((psi1 - psi2 + psi3).dag) ==
            '<Psi_1|_(q_1) - <Psi_2|_(q_1) + <Psi_3|_(q_1)')
    assert (ascii(bra_psi1 - bra_psi2 + bra_psi3) ==
            '<Psi_1|_(q_1) - <Psi_2|_(q_1) + <Psi_3|_(q_1)')
    assert ascii((psi1 * phi).dag) == '<Psi_1|_(q_1) * <Phi|_(q_2)'
    assert ascii(bra_psi1 * bra_phi) == '<Psi_1|_(q_1) * <Phi|_(q_2)'
    assert (bra_psi1_l * bra_phi_l).label == 'Psi_1,Phi'
    assert ascii(bra_psi1_l * bra_phi_l) == '<Psi_1,Phi|_(q_1*q_2)'
    assert ascii(Bra(phase * psi1)) == 'exp(I*gamma) * <Psi_1|_(q_1)'
    assert ascii((A * psi1).dag) == '<Psi_1|_(q_1) A_0^(q_1)H'


def test_ascii_sop_elements():
    """Test the ascii representation of "atomic" Superoperators"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    assert ascii(SuperOperatorSymbol("A", hs=hs1)) == 'A^(q1)'
    assert ascii(SuperOperatorSymbol("A_1", hs=hs1*hs2)) == 'A_1^(q1*q2)'
    assert (ascii(SuperOperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            'Xi_2^(q1*q2)')
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
    assert ascii(A - B + C) == 'A^(q_1) + C^(q_2) - B^(q_1)'
    assert (ascii(2 * A - sqrt(gamma) * (B + C)) ==
            '2 * A^(q_1) - sqrt(gamma) * (B^(q_1) + C^(q_2))')
    assert ascii(SPre(A_op)) == 'SPre(A^(1))'
    assert ascii(SPost(A_op)) == 'SPost(A^(1))'
    assert (ascii(SuperOperatorTimesOperator(L, A_op)) == 'L^(1)[A^(1)]')
    assert (ascii(SuperOperatorTimesOperator(L, sqrt(gamma) * A_op)) ==
            'L^(1)[sqrt(gamma) * A^(1)]')
    assert (ascii(SuperOperatorTimesOperator((L + 2*M), A_op)) ==
            '(L^(1) + 2 * M^(1))[A^(1)]')
