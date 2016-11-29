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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################

import pytest

from sympy import symbols, sqrt, exp, I

from qnet.algebra.circuit_algebra import(
        CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
        Feedback, SeriesInverse)
from qnet.algebra.operator_algebra import(
        OperatorSymbol, IdentityOperator, ZeroOperator, Create, Destroy, Jz,
        Jplus, Jminus, Phase, Displace, Squeeze, LocalSigma, tr, Adjoint,
        PseudoInverse, NullSpaceProjector)
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
from qnet.printing import tex, LaTeXPrinter


def test_tex_render_string():
    """Test rendering of ascii to latex strings"""
    assert LaTeXPrinter.render_string('a') == r'a'
    assert LaTeXPrinter.render_string('A') == r'A'
    assert LaTeXPrinter.render_string('longword') == r'\text{longword}'
    assert LaTeXPrinter.render_string('alpha') == r'\alpha'
    assert LaTeXPrinter.render_string('Alpha') == r'A'
    assert LaTeXPrinter.render_string('Beta') == r'B'
    assert LaTeXPrinter.render_string('Gamma') == r'\Gamma'
    assert LaTeXPrinter.render_string('Delta') == r'\Delta'
    assert LaTeXPrinter.render_string('Epsilon') == r'E'
    assert LaTeXPrinter.render_string('Zeta') == r'Z'
    assert LaTeXPrinter.render_string('Eta') == r'H'
    assert LaTeXPrinter.render_string('Theta') == r'\Theta'
    assert LaTeXPrinter.render_string('Iota') == r'I'
    assert LaTeXPrinter.render_string('Kappa') == r'K'
    assert LaTeXPrinter.render_string('Lambda') == r'\Lambda'
    assert LaTeXPrinter.render_string('Mu') == r'M'
    assert LaTeXPrinter.render_string('Nu') == r'N'
    assert LaTeXPrinter.render_string('Xi') == r'\Xi'
    assert LaTeXPrinter.render_string('Omicron') == r'O'
    assert LaTeXPrinter.render_string('Pi') == r'\Pi'
    assert LaTeXPrinter.render_string('Rho') == r'P'
    assert LaTeXPrinter.render_string('Sigma') == r'\Sigma'
    assert LaTeXPrinter.render_string('Tau') == r'T'
    assert LaTeXPrinter.render_string('Ypsilon') == r'\Upsilon'
    assert LaTeXPrinter.render_string('Upsilon') == r'\Upsilon'
    assert LaTeXPrinter.render_string('ypsilon') == r'\upsilon'
    assert LaTeXPrinter.render_string('upsilon') == r'\upsilon'
    assert LaTeXPrinter.render_string('Phi') == r'\Phi'
    assert LaTeXPrinter.render_string('Chi') == r'X'
    assert LaTeXPrinter.render_string('Psi') == r'\Psi'
    assert LaTeXPrinter.render_string('Omega') == r'\Omega'
    assert LaTeXPrinter.render_string('xi_1') == r'\xi_{1}'
    assert LaTeXPrinter.render_string('xi_1^2') == r'\xi^{2}_{1}'
    assert LaTeXPrinter.render_string('Xi_1') == r'\Xi_{1}'
    assert LaTeXPrinter.render_string('Xi_long') == r'\Xi_{\text{long}}'
    assert LaTeXPrinter.render_string('Xi_1+2') == r'\Xi_{1+2}'
    assert LaTeXPrinter.render_string('Lambda_i,j') == r'\Lambda_{i,j}'
    assert LaTeXPrinter.render_string('epsilon_mu,nu') == r'\epsilon_{\mu,\nu}'


def test_tex_circuit_elements():
    """Test the tex representation of "atomic" circuit algebra elements"""
    assert tex(CircuitSymbol("C", cdim=2)) == 'C'
    assert tex(CircuitSymbol("C_1", cdim=2)) == 'C_{1}'
    assert tex(CircuitSymbol("Xi_2", 2)) == r'\Xi_{2}'
    assert tex(CircuitSymbol("Xi_full", 2)) == r'\Xi_{\text{full}}'
    assert tex(CIdentity) == r'{\rm cid}(1)'
    assert tex(CircuitZero) == r'{\rm cid}(0)'


def test_tex_circuit_operations():
    """Test the tex representation of circuit algebra operations"""
    A = CircuitSymbol("A_test", cdim=2)
    B = CircuitSymbol("B_test", cdim=2)
    C = CircuitSymbol("C_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    perm = CPermutation.create((2, 1, 0, 3))

    assert (tex(A << B << C) ==
            'A_{\\text{test}} \\lhd B_{\\text{test}} \\lhd C_{\\text{test}}')
    assert (tex(A + B + C) ==
            r'A_{\text{test}} \boxplus B_{\text{test}} '
            r'\boxplus C_{\text{test}}')
    assert (tex(A << (beta + gamma)) ==
            r'A_{\text{test}} \lhd \left(\beta \boxplus \gamma\right)')
    assert (tex(A + (B << C)) ==
            r'A_{\text{test}} \boxplus \left(B_{\text{test}} \lhd '
            r'C_{\text{test}}\right)')
    assert (tex(perm) ==
            r'\mathbf{P}_{\sigma}\begin{pmatrix} 0 & 1 & 2 & 3 \\ '
            r'2 & 1 & 0 & 3 \end{pmatrix}')
    assert (tex(SeriesProduct(perm, (A+B))) ==
            r'\mathbf{P}_{\sigma}\begin{pmatrix} 0 & 1 & 2 & 3 \\ '
            r'2 & 1 & 0 & 3 \end{pmatrix} '
            r'\lhd \left(A_{\text{test}} \boxplus B_{\text{test}}\right)')
    assert (tex(Feedback((A+B), out_port=3, in_port=0)) ==
            r'\left\lfloor{A_{\text{test}} \boxplus B_{\text{test}}}'
            r'\right\rfloor_{3\rightarrow{}0}')
    assert (tex(SeriesInverse(A+B)) ==
            r'\left[A_{\text{test}} \boxplus B_{\text{test}}\right]^{\rhd}')


def test_tex_hilbert_elements():
    """Test the tex representation of "atomic" Hilbert space algebra
    elements"""
    assert tex(LocalSpace(1)) == r'\mathcal{H}_{1}'
    assert tex(LocalSpace(1, dimension=2)) == r'\mathcal{H}_{1}'
    assert tex(LocalSpace(1, basis=(r'g', 'e'))) == r'\mathcal{H}_{1}'
    assert tex(LocalSpace('local')) == r'\mathcal{H}_{\text{local}}'
    assert tex(LocalSpace('kappa')) == r'\mathcal{H}_{\kappa}'
    assert tex(TrivialSpace) == r'\mathcal{H}_{\text{null}}'
    assert tex(FullSpace) == r'\mathcal{H}_{\text{total}}'


def test_tex_hilbert_operations():
    """Test the tex representation of Hilbert space algebra operations"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    assert tex(H1 * H2) == r'\mathcal{H}_{1} \otimes \mathcal{H}_{2}'


def test_tex_matrix():
    """Test tex representation of the Matrix class"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    assert tex(OperatorSymbol("A", hs=1)) == r'\hat{A}^{(1)}'
    assert (tex(Matrix([[A, B], [C, D]])) ==
            r'\begin{pmatrix}\hat{A}^{(1)} & \hat{B}^{(1)} \\'
            r'\hat{C}^{(1)} & \hat{D}^{(1)}\end{pmatrix}')
    assert (tex(Matrix([A, B, C, D])) ==
            r'\begin{pmatrix}\hat{A}^{(1)} \\\hat{B}^{(1)} \\'
            r'\hat{C}^{(1)} \\\hat{D}^{(1)}\end{pmatrix}')
    assert (tex(Matrix([[A, B, C, D]])) ==
            r'\begin{pmatrix}\hat{A}^{(1)} & \hat{B}^{(1)} & '
            r'\hat{C}^{(1)} & \hat{D}^{(1)}\end{pmatrix}')
    assert (tex(Matrix([[0, 1], [-1, 0]])) ==
            r'\begin{pmatrix}0 & 1 \\-1 & 0\end{pmatrix}')
    assert tex(Matrix([[], []])) == r'\begin{pmatrix} \\\end{pmatrix}'
    assert tex(Matrix([])) == r'\begin{pmatrix} \\\end{pmatrix}'


def test_tex_operator_elements():
    """Test the tex representation of "atomic" operator algebra elements"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    assert tex(OperatorSymbol("A", hs=hs1)) == r'\hat{A}^{(q_{1})}'
    assert (tex(OperatorSymbol("A_1", hs=hs1*hs2)) ==
            r'\hat{A}_{1}^{(q_{1} \otimes q_{2})}')
    assert (tex(OperatorSymbol("Xi_2", hs=(r'q1', 'q2'))) ==
            r'\hat{\Xi}_{2}^{(q_{1} \otimes q_{2})}')
    assert (tex(OperatorSymbol("Xi_full", hs=1)) ==
            r'\hat{\Xi}_{\text{full}}^{(1)}')
    assert tex(IdentityOperator) == r'\mathbb{1}'
    assert tex(ZeroOperator) == r'\mathbb{0}'
    assert tex(Create(hs=1)) == r'\hat{a}^{(1)\dagger}'
    assert tex(Create(hs=1, identifier=r'b')) == r'\hat{b}^{(1)\dagger}'
    assert tex(Destroy(hs=1)) == r'\hat{a}^{(1)}'
    assert tex(Destroy(hs=1, identifier=r'b')) == r'\hat{b}^{(1)}'
    assert tex(Jz(hs=1)) == r'\hat{J}_{z}^{(1)}'
    assert tex(Jz(hs=1, identifier='Z')) == r'\hat{Z}^{(1)}'
    assert tex(Jplus(hs=1, identifier='Jp')) == r'\hat{Jp}^{(1)}'
    assert tex(Jminus(hs=1, identifier='Jm')) == r'\hat{Jm}^{(1)}'
    assert (tex(Phase(0.5, hs=1)) ==
            r'\text{Phase}^{(1)}\left(\frac{1}{2}\right)')
    assert (tex(Phase(0.5, hs=1, identifier=r'Ph')) ==
            r'\hat{Ph}^{(1)}\left(\frac{1}{2}\right)')
    assert (tex(Displace(0.5, hs=1)) ==
            r'\hat{D}^{(1)}\left(\frac{1}{2}\right)')
    assert (tex(Squeeze(0.5, hs=1)) ==
            r'\text{Squeeze}^{(1)}\left(\frac{1}{2}\right)')
    assert tex(LocalSigma('e', 'g', hs=1)) == r'\hat{\sigma}_{e,g}^{(1)}'
    assert (tex(LocalSigma('excited', 'ground', hs=1)) ==
            r'\hat{\sigma}_{\text{excited},\text{ground}}^{(1)}')
    assert (tex(LocalSigma('mu', 'nu', hs=1)) ==
            r'\hat{\sigma}_{\mu,\nu}^{(1)}')
    assert (tex(LocalSigma('excited', 'excited', hs=1)) ==
            r'\hat{\Pi}_{\text{excited}}^{(1)}')
    assert tex(LocalSigma('e', 'e', hs=1)) == r'\hat{\Pi}_{e}^{(1)}'


def test_tex_operator_operations():
    """Test the tex representation of operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    gamma = symbols('gamma', positive=True)
    assert tex(A + B) == r'\hat{A}^{(q_{1})} + \hat{B}^{(q_{1})}'
    assert tex(A * B) == r'\hat{A}^{(q_{1})} \hat{B}^{(q_{1})}'
    assert tex(A * C) == r'\hat{A}^{(q_{1})} \otimes \hat{C}^{(q_{2})}'
    assert tex(2 * A) == r'2 \hat{A}^{(q_{1})}'
    assert tex(2j * A) == r'2 i \hat{A}^{(q_{1})}'
    assert tex((1+2j) * A) == r'\left(1 + 2 i\right) \hat{A}^{(q_{1})}'
    assert tex(gamma**2 * A) == r'\gamma^{2} \hat{A}^{(q_{1})}'
    assert tex(-gamma**2/2 * A) == r'- \frac{\gamma^{2}}{2} \hat{A}^{(q_{1})}'
    assert (tex(tr(A * C, over_space=hs2)) ==
            r'{\rm tr}_{q_{2}}\left[\hat{C}^{(q_{2})}\right] '
            r'\otimes \hat{A}^{(q_{1})}')
    assert tex(Adjoint(A)) == r'\hat{A}^{(q_{1})\dagger}'
    assert tex(Adjoint(Create(hs=1))) == r'\hat{a}^{(1)}'
    assert (tex(Adjoint(A + B)) ==
            r'\left(\hat{A}^{(q_{1})} + \hat{B}^{(q_{1})}\right)^{\dagger}')
    assert tex(PseudoInverse(A)) == r'\left(\hat{A}^{(q_{1})}\right)^{+}'
    assert (tex(NullSpaceProjector(A)) ==
            r'\hat{P}_{{Ker}}\left(\hat{A}^{(q_{1})}\right)')
    assert tex(A - B) == r'\hat{A}^{(q_{1})} - \hat{B}^{(q_{1})}'
    assert (tex(A - B + C) ==
            r'\hat{A}^{(q_{1})} + \hat{C}^{(q_{2})} - \hat{B}^{(q_{1})}')
    assert (tex(2 * A - sqrt(gamma) * (B + C)) ==
            r'2 \hat{A}^{(q_{1})} - \sqrt{\gamma} \left(\hat{B}^{(q_{1})} + '
            r'\hat{C}^{(q_{2})}\right)')


def test_tex_ket_elements():
    """Test the tex representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    assert (tex(KetSymbol('Psi', hs=hs1)) ==
            r'\left|\Psi\right\rangle_{(q_{1})}')
    assert (tex(KetSymbol('Psi', hs=1)) ==
            r'\left|\Psi\right\rangle_{(1)}')
    assert (tex(KetSymbol('Psi', hs=(1, 2))) ==
            r'\left|\Psi\right\rangle_{(1 \otimes 2)}')
    assert (tex(KetSymbol('Psi', hs=hs1*hs2)) ==
            r'\left|\Psi\right\rangle_{(q_{1} \otimes q_{2})}')
    assert (tex(LocalKet('Psi', hs=1)) ==
            r'\left|\Psi\right\rangle_{(1)}')
    assert tex(ZeroKet) == '0'
    assert tex(TrivialKet) == '1'
    assert (tex(BasisKet('e', hs=hs1)) ==
            r'\left|e\right\rangle_{(q_{1})}')
    assert (tex(BasisKet('excited', hs=1)) ==
            r'\left|\text{excited}\right\rangle_{(1)}')
    assert (tex(BasisKet(1, hs=1)) ==
            r'\left|1\right\rangle_{(1)}')
    assert (tex(CoherentStateKet(2.0, hs=1)) ==
            r'\left|\alpha=2\right\rangle_{(1)}')


def test_tex_bra_elements():
    """Test the tex representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    assert (tex(Bra(KetSymbol('Psi', hs=hs1))) ==
            r'\left\langle{}\Psi\right|_{(q_{1})}')
    assert (tex(Bra(KetSymbol('Psi', hs=1))) ==
            r'\left\langle{}\Psi\right|_{(1)}')
    assert (tex(Bra(KetSymbol('Psi', hs=(1, 2)))) ==
            r'\left\langle{}\Psi\right|_{(1 \otimes 2)}')
    assert (tex(Bra(KetSymbol('Psi', hs=hs1*hs2))) ==
            r'\left\langle{}\Psi\right|_{(q_{1} \otimes q_{2})}')
    assert (tex(LocalKet('Psi', hs=1).dag) ==
            r'\left\langle{}\Psi\right|_{(1)}')
    assert tex(Bra(ZeroKet)) == '0'
    assert tex(Bra(TrivialKet)) == '1'
    assert (tex(BasisKet('e', hs=hs1).adjoint()) ==
            r'\left\langle{}e\right|_{(q_{1})}')
    assert (tex(BasisKet(1, hs=1).adjoint()) ==
            r'\left\langle{}1\right|_{(1)}')
    assert (tex(CoherentStateKet(2.0, hs=1).dag) ==
            r'\left\langle{}\alpha=2\right|_{(1)}')


def test_tex_ket_operations():
    """Test the tex representation of ket operations"""
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
    assert (tex(psi1 + psi2) ==
            r'\left|\Psi_{1}\right\rangle_{(q_{1})} + '
            r'\left|\Psi_{2}\right\rangle_{(q_{1})}')
    assert (tex(psi1 - psi2 + psi3) ==
            r'\left|\Psi_{1}\right\rangle_{(q_{1})} + '
            r'\left|\Psi_{3}\right\rangle_{(q_{1})} - '
            r'\left|\Psi_{2}\right\rangle_{(q_{1})}')
    assert (tex(psi1 * phi) ==
            r'\left|\Psi_{1}\right\rangle_{(q_{1})} \otimes '
            r'\left|\Phi\right\rangle_{(q_{2})}')
    assert (tex(psi1_l * phi_l) ==
            r'\left|\Psi_{1},\Phi\right\rangle_{(q_{1} \otimes q_{2})}')
    assert (tex(phase * psi1) ==
            r'e^{- i \gamma} \left|\Psi_{1}\right\rangle_{(q_{1})}')
    assert (tex(A * psi1) ==
            r'\hat{A}_{0}^{(q_{1})} \left|\Psi_{1}\right\rangle_{(q_{1})}')
    assert (tex(BraKet(psi1, psi2)) ==
            r'\left\langle{}\Psi_{1}\right|'
            r'\left.\Psi_{2}\right\rangle_{(q_{1})}')
    assert tex(ket_e1.dag * ket_e1) == r'1'
    assert tex(ket_g1.dag * ket_e1) == r'0'
    assert (tex(KetBra(psi1, psi2)) ==
            r'\left|\Psi_{1}\right\rangle'
            r'\left\langle{}\Psi_{2}\right|_{(q_{1})}')
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    assert (tex(bell1) ==
            r'\frac{\sqrt{2}}{2} \left('
            r'\left|e,g\right\rangle_{(q_{1} \otimes q_{2})} - '
            r'i \left|g,e\right\rangle_{(q_{1} \otimes q_{2})}\right)')
    assert (tex(bell2) ==
            r'\frac{\sqrt{2}}{2} \left('
            r'\left|e,e\right\rangle_{(q_{1} \otimes q_{2})} - '
            r'\left|g,g\right\rangle_{(q_{1} \otimes q_{2})}\right)')
    assert (tex(BraKet.create(bell1, bell2)) ==
            r'\frac{1}{2} \left('
            r'\left(\left\langle{}e,g\right|_{(q_{1} \otimes q_{2})} + '
            r'i \left\langle{}g,e\right|_{(q_{1} \otimes q_{2})}\right)'
            r'\left(\left|e,e\right\rangle_{(q_{1} \otimes q_{2})} - '
            r'\left|g,g\right\rangle_{(q_{1} \otimes q_{2})}\right)\right)')
    assert (tex(KetBra.create(bell1, bell2)) ==
            r'\frac{1}{2} \left('
            r'\left(\left|e,g\right\rangle_{(q_{1} \otimes q_{2})} - '
            r'i \left|g,e\right\rangle_{(q_{1} \otimes q_{2})}\right)'
            r'\left(\left\langle{}e,e\right|_{(q_{1} \otimes q_{2})} - '
            r'\left\langle{}g,g\right|_{(q_{1} \otimes q_{2})}\right)\right)')


def test_tex_bra_operations():
    """Test the tex representation of bra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
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
    assert (tex((psi1 + psi2).dag) ==
            r'\left\langle{}\Psi_{1}\right|_{(q_{1})} + '
            r'\left\langle{}\Psi_{2}\right|_{(q_{1})}')
    assert (tex(bra_psi1 + bra_psi2) ==
            r'\left\langle{}\Psi_{1}\right|_{(q_{1})} + '
            r'\left\langle{}\Psi_{2}\right|_{(q_{1})}')
    assert (tex(bra_psi1 - bra_psi2 + bra_psi3) ==
            r'\left\langle{}\Psi_{1}\right|_{(q_{1})} + '
            r'\left\langle{}\Psi_{3}\right|_{(q_{1})} - '
            r'\left\langle{}\Psi_{2}\right|_{(q_{1})}')
    assert (tex(bra_psi1 * bra_phi) ==
            r'\left\langle{}\Psi_{1}\right|_{(q_{1})} \otimes '
            r'\left\langle{}\Phi\right|_{(q_{2})}')
    assert (tex(bra_psi1_l * bra_phi_l) ==
            r'\left\langle{}\Psi_{1},\Phi\right|_{(q_{1} \otimes q_{2})}')
    assert (tex(Bra(phase * psi1)) ==
            r'e^{i \gamma} \left\langle{}\Psi_{1}\right|_{(q_{1})}')
    assert (tex((A * psi1).dag) ==
            r'\left\langle{}\Psi_{1}\right|_{(q_{1})} '
            r'\hat{A}_{0}^{(q_{1})\dagger}')


def test_tex_sop_elements():
    """Test the tex representation of "atomic" Superoperators"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    assert tex(SuperOperatorSymbol("A", hs=hs1)) == r'\mathrm{A}^{(q_{1})}'
    assert (tex(SuperOperatorSymbol("A_1", hs=hs1*hs2)) ==
            r'\mathrm{A}_{1}^{(q_{1} \otimes q_{2})}')
    assert (tex(SuperOperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            r'\mathrm{\Xi}_{2}^{(q_{1} \otimes q_{2})}')
    assert (tex(SuperOperatorSymbol("Xi_full", hs=1)) ==
            r'\mathrm{\Xi}_{\text{full}}^{(1)}')
    assert tex(IdentitySuperOperator) == r'\mathbb{1}'
    assert tex(ZeroSuperOperator) == r'\mathbb{0}'


def test_tex_sop_operations():
    """Test the tex representation of super operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = SuperOperatorSymbol("A", hs=hs1)
    B = SuperOperatorSymbol("B", hs=hs1)
    C = SuperOperatorSymbol("C", hs=hs2)
    L = SuperOperatorSymbol("L", hs=1)
    M = SuperOperatorSymbol("M", hs=1)
    A_op = OperatorSymbol("A", hs=1)
    gamma = symbols('gamma', positive=True)
    assert tex(A + B) == r'\mathrm{A}^{(q_{1})} + \mathrm{B}^{(q_{1})}'
    assert tex(A * B) == r'\mathrm{A}^{(q_{1})} \mathrm{B}^{(q_{1})}'
    assert tex(A * C) == r'\mathrm{A}^{(q_{1})} \otimes \mathrm{C}^{(q_{2})}'
    assert tex(2 * A) == r'2 \mathrm{A}^{(q_{1})}'
    assert tex(2j * A) == r'2 i \mathrm{A}^{(q_{1})}'
    assert tex((1+2j) * A) == r'\left(1 + 2 i\right) \mathrm{A}^{(q_{1})}'
    assert tex(gamma**2 * A) == r'\gamma^{2} \mathrm{A}^{(q_{1})}'
    assert (tex(-gamma**2/2 * A) ==
            r'- \frac{\gamma^{2}}{2} \mathrm{A}^{(q_{1})}')
    assert tex(SuperAdjoint(A)) == r'\mathrm{A}^{(q_{1})\dagger}'
    assert (tex(SuperAdjoint(A + B)) ==
            r'\left(\mathrm{A}^{(q_{1})} + '
            r'\mathrm{B}^{(q_{1})}\right)^{\dagger}')
    assert tex(A - B) == r'\mathrm{A}^{(q_{1})} - \mathrm{B}^{(q_{1})}'
    assert (tex(A - B + C) ==
            r'\mathrm{A}^{(q_{1})} + \mathrm{C}^{(q_{2})} - '
            r'\mathrm{B}^{(q_{1})}')
    assert (tex(2 * A - sqrt(gamma) * (B + C)) ==
            r'2 \mathrm{A}^{(q_{1})} - \sqrt{\gamma} '
            r'\left(\mathrm{B}^{(q_{1})} + \mathrm{C}^{(q_{2})}\right)')
    assert tex(SPre(A_op)) == r'{\rm SPre\left(\hat{A}^{(1)}\right)}'
    assert tex(SPost(A_op)) == r'{\rm SPost\left(\hat{A}^{(1)}\right)}'
    assert (tex(SuperOperatorTimesOperator(L, A_op)) ==
            r'\mathrm{L}^{(1)}\left[\hat{A}^{(1)}\right]')
    assert (tex(SuperOperatorTimesOperator(L, sqrt(gamma) * A_op)) ==
            r'\mathrm{L}^{(1)}\left[\sqrt{\gamma} \hat{A}^{(1)}\right]')
    assert (tex(SuperOperatorTimesOperator((L + 2*M), A_op)) ==
            r'\left(\mathrm{L}^{(1)} + 2 \mathrm{M}^{(1)}\right)'
            r'\left[\hat{A}^{(1)}\right]')
