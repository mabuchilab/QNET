from functools import partial

import pytest

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
    SuperOperatorTimesOperator, FockIndex, StrLabel, IdxSym, latex,
    configure_printing, QuantumDerivative, Scalar, ScalarExpression, SpinSpace,
    SpinBasisKet, Eq)
from qnet.printing.latexprinter import QnetLatexPrinter


def test_ascii_scalar():
    """Test rendering of scalar values"""
    assert latex(2) == '2'
    latex.printer.cache = {}
    # we always want 2.0 to be printed as '2'. Without this normalization, the
    # state of the cache might introduce non-reproducible behavior, as 2==2.0
    assert latex(2.0) == '2'
    assert latex(1j) == '1i'
    assert latex('foo') == 'foo'

    i = IdxSym('i')
    alpha = IndexedBase('alpha')
    assert latex(i) == 'i'
    assert latex(alpha[i]) == r'\alpha_{i}'


def test_tex_render_string():
    """Test rendering of ascii to latex strings"""
    printer = QnetLatexPrinter()
    assert printer._render_str('a') == r'a'
    assert printer._render_str('A') == r'A'
    assert printer._render_str('longword') == r'\text{longword}'
    assert printer._render_str('alpha') == r'\alpha'
    assert latex('alpha') == r'\alpha'
    assert printer._render_str('Alpha') == r'A'
    assert printer._render_str('Beta') == r'B'
    assert printer._render_str('Gamma') == r'\Gamma'
    assert printer._render_str('Delta') == r'\Delta'
    assert printer._render_str('Epsilon') == r'E'
    assert printer._render_str('Zeta') == r'Z'
    assert printer._render_str('Eta') == r'H'
    assert printer._render_str('Theta') == r'\Theta'
    assert printer._render_str('Iota') == r'I'
    assert printer._render_str('Kappa') == r'K'
    assert printer._render_str('Lambda') == r'\Lambda'
    assert printer._render_str('Mu') == r'M'
    assert printer._render_str('Nu') == r'N'
    assert printer._render_str('Xi') == r'\Xi'
    assert printer._render_str('Omicron') == r'O'
    assert printer._render_str('Pi') == r'\Pi'
    assert printer._render_str('Rho') == r'P'
    assert printer._render_str('Sigma') == r'\Sigma'
    assert printer._render_str('Tau') == r'T'
    assert printer._render_str('Ypsilon') == r'\Upsilon'
    assert printer._render_str('Upsilon') == r'\Upsilon'
    assert printer._render_str('ypsilon') == r'\upsilon'
    assert printer._render_str('upsilon') == r'\upsilon'
    assert printer._render_str('Phi') == r'\Phi'
    assert printer._render_str('Chi') == r'X'
    assert printer._render_str('Psi') == r'\Psi'
    assert printer._render_str('Omega') == r'\Omega'
    assert printer._render_str('xi_1') == r'\xi_{1}'
    assert printer._render_str('xi_1^2') == r'\xi_{1}^{2}'
    assert printer._render_str('Xi_1') == r'\Xi_{1}'
    assert printer._render_str('Xi_long') == r'\Xi_{\text{long}}'
    assert printer._render_str('Xi_1+2') == r'\Xi_{1+2}'
    assert printer._render_str('Lambda_i,j') == r'\Lambda_{i,j}'
    assert printer._render_str('epsilon_mu,nu') == r'\epsilon_{\mu,\nu}'


def test_tex_circuit_elements():
    """Test the tex representation of "atomic" circuit algebra elements"""
    alpha, t = symbols('alpha, t')
    theta = symbols('theta', positive=True)
    assert latex(CircuitSymbol("C", cdim=2)) == 'C'
    assert latex(CircuitSymbol("C_1", cdim=2)) == 'C_{1}'
    assert latex(CircuitSymbol("Xi_2", cdim=2)) == r'\Xi_{2}'
    assert latex(CircuitSymbol("Xi_full", cdim=2)) == r'\Xi_{\text{full}}'
    assert (
        latex(CircuitSymbol("C", alpha, t, cdim=2)) ==
        r'C\left(\alpha, t\right)')
    assert latex(CIdentity) == r'{\rm cid}(1)'
    assert latex(cid(4)) == r'{\rm cid}(4)'
    assert latex(CircuitZero) == r'{\rm cid}(0)'
    assert latex(Beamsplitter()) == r'{\rm BS}\left(\frac{\pi}{4}\right)'
    assert (
        latex(Beamsplitter(mixing_angle=theta)) ==
        r'{\rm BS}\left(\theta\right)')


def test_tex_circuit_operations():
    """Test the tex representation of circuit algebra operations"""
    A = CircuitSymbol("A_test", cdim=2)
    B = CircuitSymbol("B_test", cdim=2)
    C = CircuitSymbol("C_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    perm = CPermutation.create((2, 1, 0, 3))

    assert (latex(A << B << C) ==
            r'A_{\text{test}} \lhd B_{\text{test}} \lhd C_{\text{test}}')
    assert (latex(A + B + C) ==
            r'A_{\text{test}} \boxplus B_{\text{test}} '
            r'\boxplus C_{\text{test}}')
    assert (latex(A << (beta + gamma)) ==
            r'A_{\text{test}} \lhd \left(\beta \boxplus \gamma\right)')
    assert (latex(A + (B << C)) ==
            r'A_{\text{test}} \boxplus '
            r'\left(B_{\text{test}} \lhd C_{\text{test}}\right)')
    assert (latex(perm) ==
            r'\mathbf{P}_{\sigma}\begin{pmatrix} 0 & 1 & 2 & 3 \\ '
            r'2 & 1 & 0 & 3 \end{pmatrix}')
    assert (latex(SeriesProduct(perm, (A+B))) ==
            r'\mathbf{P}_{\sigma}\begin{pmatrix} 0 & 1 & 2 & 3 \\ '
            r'2 & 1 & 0 & 3 \end{pmatrix} '
            r'\lhd \left(A_{\text{test}} \boxplus B_{\text{test}}\right)')
    assert (latex(Feedback((A+B), out_port=3, in_port=0)) ==
            r'\left\lfloor{A_{\text{test}} \boxplus B_{\text{test}}}'
            r'\right\rfloor_{3\rightarrow{}0}')
    assert (latex(SeriesInverse(A+B)) ==
            r'\left[A_{\text{test}} \boxplus B_{\text{test}}\right]^{\rhd}')


def test_tex_hilbert_elements():
    """Test the tex representation of "atomic" Hilbert space algebra
    elements"""
    assert latex(LocalSpace(1)) == r'\mathcal{H}_{1}'
    assert latex(LocalSpace(1, dimension=2)) == r'\mathcal{H}_{1}'
    assert latex(LocalSpace(1, basis=(r'g', 'e'))) == r'\mathcal{H}_{1}'
    assert latex(LocalSpace('local')) == r'\mathcal{H}_{\text{local}}'
    assert latex(LocalSpace('kappa')) == r'\mathcal{H}_{\kappa}'
    assert latex(TrivialSpace) == r'\mathcal{H}_{\text{null}}'
    assert latex(FullSpace) == r'\mathcal{H}_{\text{total}}'
    assert latex(LocalSpace(StrLabel(IdxSym('i')))) == r'\mathcal{H}_{i}'


def test_tex_hilbert_operations():
    """Test the tex representation of Hilbert space algebra operations"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    assert latex(H1 * H2) == r'\mathcal{H}_{1} \otimes \mathcal{H}_{2}'


def test_tex_matrix():
    """Test tex representation of the Matrix class"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)
    assert latex(OperatorSymbol("A", hs=1)) == r'\hat{A}^{(1)}'
    assert (latex(Matrix([[A, B], [C, D]])) ==
            r'\begin{pmatrix}\hat{A}^{(1)} & \hat{B}^{(1)} \\'
            r'\hat{C}^{(1)} & \hat{D}^{(1)}\end{pmatrix}')
    assert (latex(Matrix([A, B, C, D])) ==
            r'\begin{pmatrix}\hat{A}^{(1)} \\\hat{B}^{(1)} \\'
            r'\hat{C}^{(1)} \\\hat{D}^{(1)}\end{pmatrix}')
    assert (latex(Matrix([[A, B, C, D]])) ==
            r'\begin{pmatrix}\hat{A}^{(1)} & \hat{B}^{(1)} & '
            r'\hat{C}^{(1)} & \hat{D}^{(1)}\end{pmatrix}')
    assert (latex(Matrix([[0, 1], [-1, 0]])) ==
            r'\begin{pmatrix}0 & 1 \\-1 & 0\end{pmatrix}')
    assert latex(Matrix([[], []])) == r'\begin{pmatrix} \\\end{pmatrix}'
    assert latex(Matrix([])) == r'\begin{pmatrix} \\\end{pmatrix}'


def test_tex_equation():
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
    assert (
        latex(eq_1).split("\n") == [
            r'\begin{equation}',
            r'  \hat{H}^{(0)} = \hat{a}^{(0)\dagger} \hat{a}^{(0)}',
            r'\end{equation}',
            ''])
    assert (
        latex(eq_1.set_tag(1)).split("\n") == [
            r'\begin{equation}',
            r'  \hat{H}^{(0)} = \hat{a}^{(0)\dagger} \hat{a}^{(0)}',
            r'\tag{1}\end{equation}',
            ''])
    tex_lines = (
        latex(eq, show_hs_label=False, tex_op_macro=r'\Op{{{name}}}')
        .split("\n"))
    expected = [
        r'\begin{align}',
        r'  \Op{H} &= \Op{a}^{\dagger} \Op{a}\\',
        r'  \mathbb{1} + \Op{H} &= \Op{a}^{\dagger} \Op{a}\\',
        r'   &= \mathbb{1} + \Op{a}^{\dagger} \Op{a}\\',
        r'   &= \left(\mathbb{1} + \Op{a}^{\dagger} \Op{a}\right) \left(\mathbb{1} + \Op{a}^{\dagger} \Op{a}\right)\tag{3}\\',
        r'  2 + \Op{H} &= \mathbb{1} + \left(\mathbb{1} + \Op{a}^{\dagger} \Op{a}\right) \left(\mathbb{1} + \Op{a}^{\dagger} \Op{a}\right)\tag{4}\\',
        r'   &= 2 + \Op{a}^{\dagger} \Op{a}^{\dagger} \Op{a} \Op{a} + 3 \Op{a}^{\dagger} \Op{a}\\',
        r'  \left(2 + \Op{H}\right) \left(2 + \Op{H}\right) &= 2 + \Op{a}^{\dagger} \Op{a}^{\dagger} \Op{a} \Op{a} + 3 \Op{a}^{\dagger} \Op{a}\tag{5}\\',
        r'  4 + 4 \Op{H} + \Op{H} \Op{H} &= 2 + \Op{a}^{\dagger} \Op{a}^{\dagger} \Op{a} \Op{a} + 3 \Op{a}^{\dagger} \Op{a}\\',
        r'  \left(4 + 4 \Op{H} + \Op{H} \Op{H}\right) \left(4 + 4 \Op{H} + \Op{H} \Op{H}\right) &= 2 + \Op{a}^{\dagger} \Op{a}^{\dagger} \Op{a} \Op{a} + 3 \Op{a}^{\dagger} \Op{a}\tag{6}\\',
        r'  16 + 32 \Op{H} + \Op{H} \Op{H} \Op{H} \Op{H} + 8 \Op{H} \Op{H} \Op{H} + 24 \Op{H} \Op{H} &= 2 + \Op{a}^{\dagger} \Op{a}^{\dagger} \Op{a} \Op{a} + 3 \Op{a}^{\dagger} \Op{a}\\',
        r'   &= 3 + \Op{a}^{\dagger} \Op{a}^{\dagger} \Op{a} \Op{a} + 3 \Op{a}^{\dagger} \Op{a}',
        r'\end{align}',
        r'']
    for i, line in enumerate(tex_lines):
        assert line == expected[i]


def test_tex_operator_elements():
    """Test the tex representation of "atomic" operator algebra elements"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    alpha, beta = symbols('alpha, beta')
    fock1 = LocalSpace(
       1, local_identifiers={'Create': 'b', 'Destroy': 'b', 'Phase': 'Phi'})
    spin1 = SpinSpace(
       1, spin=1, local_identifiers={'Jz': 'Z', 'Jplus': 'Jp', 'Jminus': 'Jm'})
    assert latex(OperatorSymbol("A", hs=hs1)) == r'\hat{A}^{(q_{1})}'
    assert (latex(OperatorSymbol("A_1", hs=hs1*hs2)) ==
            r'\hat{A}_{1}^{(q_{1} \otimes q_{2})}')
    assert (latex(OperatorSymbol("Xi_2", hs=(r'q1', 'q2'))) ==
            r'\hat{\Xi}_{2}^{(q_{1} \otimes q_{2})}')
    assert (latex(OperatorSymbol("Xi_full", hs=1)) ==
            r'\hat{\Xi}_{\text{full}}^{(1)}')
    assert latex(OperatorSymbol("Xi", alpha, beta, hs=1)) == (
        r'\hat{\Xi}^{(1)}\left(\alpha, \beta\right)')
    assert latex(IdentityOperator) == r'\mathbb{1}'
    assert latex(IdentityOperator, tex_identity_sym='I') == 'I'
    assert latex(ZeroOperator) == r'\mathbb{0}'
    assert latex(Create(hs=1)) == r'\hat{a}^{(1)\dagger}'
    assert latex(Create(hs=fock1)) == r'\hat{b}^{(1)\dagger}'
    assert latex(Destroy(hs=1)) == r'\hat{a}^{(1)}'
    assert latex(Destroy(hs=fock1)) == r'\hat{b}^{(1)}'
    assert latex(Jz(hs=SpinSpace(1, spin=1))) == r'\hat{J}_{z}^{(1)}'
    assert latex(Jz(hs=spin1)) == r'\hat{Z}^{(1)}'
    assert latex(Jplus(hs=SpinSpace(1, spin=1))) == r'\hat{J}_{+}^{(1)}'
    assert latex(Jplus(hs=spin1)) == r'\text{Jp}^{(1)}'
    assert latex(Jminus(hs=SpinSpace(1, spin=1))) == r'\hat{J}_{-}^{(1)}'
    assert latex(Jminus(hs=spin1)) == r'\text{Jm}^{(1)}'
    assert (latex(Phase(Rational(1, 2), hs=1)) ==
            r'\text{Phase}^{(1)}\left(\frac{1}{2}\right)')
    assert (latex(Phase(0.5, hs=1)) ==
            r'\text{Phase}^{(1)}\left(0.5\right)')
    assert (latex(Phase(0.5, hs=fock1)) ==
            r'\hat{\Phi}^{(1)}\left(0.5\right)')
    assert (latex(Displace(0.5, hs=1)) ==
            r'\hat{D}^{(1)}\left(0.5\right)')
    assert (latex(Squeeze(0.5, hs=1)) ==
            r'\text{Squeeze}^{(1)}\left(0.5\right)')
    hs_tls = LocalSpace('1', basis=('g', 'e'))
    sig_e_g = LocalSigma('e', 'g', hs=hs_tls)
    assert (
        latex(sig_e_g, sig_as_ketbra=False) ==
        r'\hat{\sigma}_{e,g}^{(1)}')
    assert (
        latex(sig_e_g) ==
        r'\left\lvert e \middle\rangle\!\middle\langle g \right\rvert^{(1)}')
    hs_tls = LocalSpace('1', basis=('excited', 'ground'))
    sig_excited_ground = LocalSigma('excited', 'ground', hs=hs_tls)
    assert (
        latex(sig_excited_ground, sig_as_ketbra=False) ==
        r'\hat{\sigma}_{\text{excited},\text{ground}}^{(1)}')
    assert (
        latex(sig_excited_ground) ==
        r'\left\lvert \text{excited} \middle\rangle\!'
        r'\middle\langle \text{ground} \right\rvert^{(1)}')
    hs_tls = LocalSpace('1', basis=('mu', 'nu'))
    sig_mu_nu = LocalSigma('mu', 'nu', hs=hs_tls)
    assert (
        latex(sig_mu_nu) ==
        r'\left\lvert \mu \middle\rangle\!'
        r'\middle\langle \nu \right\rvert^{(1)}')
    hs_tls = LocalSpace('1', basis=('excited', 'ground'))
    sig_excited_excited = LocalProjector('excited', hs=hs_tls)
    assert (
        latex(sig_excited_excited, sig_as_ketbra=False) ==
        r'\hat{\Pi}_{\text{excited}}^{(1)}')
    hs_tls = LocalSpace('1', basis=('g', 'e'))
    sig_e_e = LocalProjector('e', hs=hs_tls)
    assert (
        latex(sig_e_e, sig_as_ketbra=False) == r'\hat{\Pi}_{e}^{(1)}')


def test_tex_operator_operations():
    """Test the tex representation of operator algebra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    psi = KetSymbol('Psi', hs=hs1)
    gamma = symbols('gamma', positive=True)
    assert latex(A.dag()) == r'\hat{A}^{(q_{1})\dagger}'
    assert latex(A + B) == r'\hat{A}^{(q_{1})} + \hat{B}^{(q_{1})}'
    assert latex(A * B) == r'\hat{A}^{(q_{1})} \hat{B}^{(q_{1})}'
    assert latex(A * C) == r'\hat{A}^{(q_{1})} \hat{C}^{(q_{2})}'
    assert latex(2 * A) == r'2 \hat{A}^{(q_{1})}'
    assert latex(2j * A) == r'2i \hat{A}^{(q_{1})}'
    assert latex((1+2j) * A) == r'(1+2i) \hat{A}^{(q_{1})}'
    assert latex(gamma**2 * A) == r'\gamma^{2} \hat{A}^{(q_{1})}'
    assert (
        latex(-gamma**2/2 * A) == r'- \frac{\gamma^{2}}{2} \hat{A}^{(q_{1})}')
    assert (
        latex(tr(A * C, over_space=hs2)) ==
        r'{\rm tr}_{q_{2}}\left[\hat{C}^{(q_{2})}\right] '
        r'\hat{A}^{(q_{1})}')
    assert latex(Adjoint(A)) == r'\hat{A}^{(q_{1})\dagger}'
    assert (
        latex(Adjoint(A**2)) ==
        r'\left(\hat{A}^{(q_{1})} \hat{A}^{(q_{1})}\right)^\dagger')
    assert (
        latex(Adjoint(A)**2) ==
        r'\hat{A}^{(q_{1})\dagger} \hat{A}^{(q_{1})\dagger}')
    assert latex(Adjoint(Create(hs=1))) == r'\hat{a}^{(1)}'
    assert (
        latex(Adjoint(A + B)) ==
        r'\left(\hat{A}^{(q_{1})} + \hat{B}^{(q_{1})}\right)^\dagger')
    assert latex(PseudoInverse(A)) == r'\left(\hat{A}^{(q_{1})}\right)^+'
    assert (
        latex(PseudoInverse(A)**2) ==
        r'\left(\hat{A}^{(q_{1})}\right)^+ \left(\hat{A}^{(q_{1})}\right)^+')
    assert (latex(NullSpaceProjector(A)) ==
            r'\hat{P}_{Ker}\left(\hat{A}^{(q_{1})}\right)')
    assert latex(A - B) == r'\hat{A}^{(q_{1})} - \hat{B}^{(q_{1})}'
    assert (latex(A - B + C) ==
            r'\hat{A}^{(q_{1})} - \hat{B}^{(q_{1})} + \hat{C}^{(q_{2})}')
    assert (latex(2 * A - sqrt(gamma) * (B + C)) ==
            r'2 \hat{A}^{(q_{1})} - \sqrt{\gamma} \left(\hat{B}^{(q_{1})} + '
            r'\hat{C}^{(q_{2})}\right)')
    assert (latex(Commutator(A, B)) ==
            r'\left[\hat{A}^{(q_{1})}, \hat{B}^{(q_{1})}\right]')
    expr = (Commutator(A, B) * psi).dag()
    assert (
        latex(expr, show_hs_label=False) ==
        r'\left\langle \Psi \right\rvert \left[\hat{A}, '
        r'\hat{B}\right]^{\dagger}')


def test_tex_ket_elements():
    """Test the tex representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    alpha, beta = symbols('alpha, beta')
    psi = KetSymbol('Psi', hs=hs1)
    assert (latex(psi) == r'\left\lvert \Psi \right\rangle^{(q_{1})}')
    assert (
        latex(KetSymbol('Psi', alpha, beta, hs=1)) ==
        r'\left\lvert \Psi\left(\alpha, \beta\right) \right\rangle^{(1)}')
    assert (latex(psi, tex_use_braket=True) == r'\Ket{\Psi}^{(q_{1})}')
    assert (
        latex(psi, tex_use_braket=True, show_hs_label='subscript') ==
        r'\Ket{\Psi}_{(q_{1})}')
    assert (
        latex(psi, tex_use_braket=True, show_hs_label=False) == r'\Ket{\Psi}')
    assert (latex(KetSymbol('Psi', hs=1)) ==
            r'\left\lvert \Psi \right\rangle^{(1)}')
    assert (latex(KetSymbol('Psi', hs=(1, 2))) ==
            r'\left\lvert \Psi \right\rangle^{(1 \otimes 2)}')
    assert (latex(KetSymbol('Psi', hs=hs1*hs2)) ==
            r'\left\lvert \Psi \right\rangle^{(q_{1} \otimes q_{2})}')
    assert (latex(KetSymbol('Psi', hs=1)) ==
            r'\left\lvert \Psi \right\rangle^{(1)}')
    assert latex(ZeroKet) == '0'
    assert latex(TrivialKet) == '1'
    assert (latex(BasisKet('e', hs=hs1)) ==
            r'\left\lvert e \right\rangle^{(q_{1})}')
    hs_tls = LocalSpace('1', basis=('excited', 'ground'))
    assert (latex(BasisKet('excited', hs=hs_tls)) ==
            r'\left\lvert \text{excited} \right\rangle^{(1)}')
    assert (latex(BasisKet(1, hs=1)) ==
            r'\left\lvert 1 \right\rangle^{(1)}')
    spin = SpinSpace('s', spin=(3, 2))
    assert (
        latex(SpinBasisKet(-3, 2, hs=spin)) ==
        r'\left\lvert -3/2 \right\rangle^{(s)}')
    assert (
        latex(SpinBasisKet(1, 2, hs=spin)) ==
        r'\left\lvert +1/2 \right\rangle^{(s)}')
    assert (
        latex(SpinBasisKet(-3, 2, hs=spin), tex_frac_for_spin_labels=True) ==
        r'\left\lvert -\frac{3}{2} \right\rangle^{(s)}')
    assert (
        latex(SpinBasisKet(1, 2, hs=spin), tex_frac_for_spin_labels=True) ==
        r'\left\lvert +\frac{1}{2} \right\rangle^{(s)}')
    assert (latex(CoherentStateKet(2.0, hs=1)) ==
            r'\left\lvert \alpha=2 \right\rangle^{(1)}')


def test_tex_symbolic_labels():
    """Test tex representation of symbols with symbolic labels"""
    i = IdxSym('i')
    j = IdxSym('j')
    hs0 = LocalSpace(0)
    hs1 = LocalSpace(1)
    Psi = IndexedBase('Psi')
    with configure_printing(tex_use_braket=True):
        assert (
            latex(BasisKet(FockIndex(2 * i), hs=hs0)) ==
            r'\Ket{2 i}^{(0)}')
        assert (latex(
            KetSymbol(StrLabel(2 * i), hs=hs0)) ==
            r'\Ket{2 i}^{(0)}')
        assert (
            latex(KetSymbol(StrLabel(Psi[i, j]), hs=hs0*hs1)) ==
            r'\Ket{\Psi_{i j}}^{(0 \otimes 1)}')
        expr = BasisKet(FockIndex(i), hs=hs0) * BasisKet(FockIndex(j), hs=hs1)
        assert latex(expr) == r'\Ket{i,j}^{(0 \otimes 1)}'
        assert (
            latex(Bra(BasisKet(FockIndex(2 * i), hs=hs0))) ==
            r'\Bra{2 i}^{(0)}')
        assert (
            latex(LocalSigma(FockIndex(i), FockIndex(j), hs=hs0)) ==
            r'\Ket{i}\!\Bra{j}^{(0)}')
        alpha = symbols('alpha')
        expr = CoherentStateKet(alpha, hs=1).to_fock_representation()
        assert (
            latex(expr) ==
            r'e^{- \frac{\alpha \overline{\alpha}}{2}} '
            r'\left(\sum_{n \in \mathcal{H}_{1}} '
            r'\frac{\alpha^{n}}{\sqrt{n!}} \Ket{n}^{(1)}\right)')
        assert (
            latex(expr, conjg_style='star') ==
            r'e^{- \frac{\alpha {\alpha}^*}{2}} '
            r'\left(\sum_{n \in \mathcal{H}_{1}} '
            r'\frac{\alpha^{n}}{\sqrt{n!}} \Ket{n}^{(1)}\right)')

    tls = SpinSpace(label='s', spin='1/2', basis=('down', 'up'))
    Sig = IndexedBase('sigma')
    n = IdxSym('n')
    Sig_n = OperatorSymbol(StrLabel(Sig[n]), hs=tls)
    assert latex(Sig_n, show_hs_label=False) == r'\hat{\sigma}_{n}'


def test_tex_bra_elements():
    """Test the tex representation of "atomic" kets"""
    hs1 = LocalSpace('q1', basis=('g', 'e'))
    hs2 = LocalSpace('q2', basis=('g', 'e'))
    alpha, beta = symbols('alpha, beta')
    bra = Bra(KetSymbol('Psi', hs=hs1))
    assert (latex(bra) == r'\left\langle \Psi \right\rvert^{(q_{1})}')
    assert latex(Bra(KetSymbol('Psi', alpha, beta, hs=hs1))) == (
        r'\left\langle \Psi\left(\alpha, \beta\right) \right\rvert^{(q_{1})}')
    assert (latex(bra, tex_use_braket=True) == r'\Bra{\Psi}^{(q_{1})}')
    assert (
        latex(bra, tex_use_braket=True, show_hs_label='subscript') ==
        r'\Bra{\Psi}_{(q_{1})}')
    assert (
        latex(bra, tex_use_braket=True, show_hs_label=False) ==
        r'\Bra{\Psi}')
    assert (
        latex(Bra(KetSymbol('Psi', hs=1))) ==
        r'\left\langle \Psi \right\rvert^{(1)}')
    assert (
        latex(Bra(KetSymbol('Psi', hs=(1, 2)))) ==
        r'\left\langle \Psi \right\rvert^{(1 \otimes 2)}')
    assert (
        latex(Bra(KetSymbol('Psi', hs=hs1*hs2))) ==
        r'\left\langle \Psi \right\rvert^{(q_{1} \otimes q_{2})}')
    assert (
        latex(KetSymbol('Psi', hs=1).dag()) ==
        r'\left\langle \Psi \right\rvert^{(1)}')
    assert latex(Bra(ZeroKet)) == '0'
    assert latex(Bra(TrivialKet)) == '1'
    assert (
        latex(BasisKet('e', hs=hs1).adjoint()) ==
        r'\left\langle e \right\rvert^{(q_{1})}')
    assert (
        latex(BasisKet(1, hs=1).adjoint()) ==
        r'\left\langle 1 \right\rvert^{(1)}')
    assert (
        latex(CoherentStateKet(2.0, hs=1).dag()) ==
        r'\left\langle \alpha=2 \right\rvert^{(1)}')


def test_tex_ket_operations():
    """Test the tex representation of ket operations"""
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
    assert (
        latex(psi1 + psi2) ==
        r'\left\lvert \Psi_{1} \right\rangle^{(q_{1})} + '
        r'\left\lvert \Psi_{2} \right\rangle^{(q_{1})}')
    assert (
        latex(psi1 - psi2 + psi3) ==
        r'\left\lvert \Psi_{1} \right\rangle^{(q_{1})} - '
        r'\left\lvert \Psi_{2} \right\rangle^{(q_{1})} + '
        r'\left\lvert \Psi_{3} \right\rangle^{(q_{1})}')
    assert (
        latex(psi1 * phi) ==
        r'\left\lvert \Psi_{1} \right\rangle^{(q_{1})} \otimes '
        r'\left\lvert \Phi \right\rangle^{(q_{2})}')
    assert (
        latex(phase * psi1) ==
        r'e^{- i \gamma} \left\lvert \Psi_{1} \right\rangle^{(q_{1})}')
    assert (
        latex((alpha + 1) * KetSymbol('Psi', hs=0)) ==
        r'\left(\alpha + 1\right) \left\lvert \Psi \right\rangle^{(0)}')
    assert (
        latex(A * psi1) ==
        r'\hat{A}_{0}^{(q_{1})} \left\lvert \Psi_{1} \right\rangle^{(q_{1})}')
    braket = BraKet(psi1, psi2)
    assert (
        latex(braket, show_hs_label='subscript') ==
        r'\left\langle \Psi_{1} \middle\vert \Psi_{2} \right\rangle_{(q_{1})}')
    assert (
        latex(braket, show_hs_label=False) ==
        r'\left\langle \Psi_{1} \middle\vert \Psi_{2} \right\rangle')
    expr = BraKet(
        KetSymbol('Psi_1', alpha, hs=hs1), KetSymbol('Psi_2', beta, hs=hs1))
    assert (
        latex(expr) ==
        r'\left\langle \Psi_{1}\left(\alpha\right) \middle\vert '
        r'\Psi_{2}\left(\beta\right) \right\rangle^{(q_{1})}')
    assert (
        latex(ket_e1 * ket_e2) ==
        r'\left\lvert ee \right\rangle^{(q_{1} \otimes q_{2})}')
    assert latex(ket_e1.dag() * ket_e1) == r'1'
    assert latex(ket_g1.dag() * ket_e1) == r'0'
    ketbra = KetBra(psi1, psi2)
    assert (
        latex(ketbra) ==
        r'\left\lvert \Psi_{1} \middle\rangle\!'
        r'\middle\langle \Psi_{2} \right\rvert^{(q_{1})}')
    assert (
        latex(ketbra, show_hs_label='subscript') ==
        r'\left\lvert \Psi_{1} \middle\rangle\!'
        r'\middle\langle \Psi_{2} \right\rvert_{(q_{1})}')
    assert (
        latex(ketbra, show_hs_label=False) ==
        r'\left\lvert \Psi_{1} \middle\rangle\!'
        r'\middle\langle \Psi_{2} \right\rvert')
    expr = KetBra(
        KetSymbol('Psi_1', alpha, hs=hs1), KetSymbol('Psi_2', beta, hs=hs1))
    assert (
        latex(expr) ==
        r'\left\lvert \Psi_{1}\left(\alpha\right) \middle\rangle\!'
        r'\middle\langle \Psi_{2}\left(\beta\right) \right\rvert^{(q_{1})}')
    bell1 = (ket_e1 * ket_g2 - I * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    assert (
        latex(bell1) ==
        r'\frac{1}{\sqrt{2}} \left(\left\lvert eg \right\rangle^{(q_{1} '
        r'\otimes q_{2})} - i \left\lvert ge \right\rangle'
        r'^{(q_{1} \otimes q_{2})}\right)')
    assert (
        latex(bell2) ==
        r'\frac{1}{\sqrt{2}} \left(\left\lvert ee \right\rangle^{(q_{1} '
        r'\otimes q_{2})} - \left\lvert gg \right\rangle'
        r'^{(q_{1} \otimes q_{2})}\right)')
    assert (
        latex(bell2, show_hs_label=False) ==
        r'\frac{1}{\sqrt{2}} \left(\left\lvert ee \right\rangle - '
        r'\left\lvert gg \right\rangle\right)')
    assert BraKet.create(bell1, bell2).expand() == 0
    assert (
        latex(BraKet.create(bell1, bell2)) ==
        r'\frac{1}{2} \left(\left\langle eg \right\rvert'
        r'^{(q_{1} \otimes q_{2})} + i \left\langle ge \right\rvert'
        r'^{(q_{1} \otimes q_{2})}\right) '
        r'\left(\left\lvert ee \right\rangle^{(q_{1} \otimes q_{2})} '
        r'- \left\lvert gg \right\rangle^{(q_{1} \otimes q_{2})}\right)')
    assert (
        latex(KetBra.create(bell1, bell2)) ==
        r'\frac{1}{2} \left(\left\lvert eg \right\rangle'
        r'^{(q_{1} \otimes q_{2})} - i \left\lvert ge \right\rangle'
        r'^{(q_{1} \otimes q_{2})}\right)\left(\left\langle ee \right\rvert'
        r'^{(q_{1} \otimes q_{2})} - \left\langle gg \right\rvert'
        r'^{(q_{1} \otimes q_{2})}\right)')
    with configure_printing(tex_use_braket=True):
        expr = KetBra(KetSymbol('Psi', hs=0), BasisKet(FockIndex(i), hs=0))
        assert latex(expr) == r'\Ket{\Psi}\!\Bra{i}^{(0)}'
        expr = KetBra(BasisKet(FockIndex(i), hs=0), KetSymbol('Psi', hs=0))
        assert latex(expr) == r'\Ket{i}\!\Bra{\Psi}^{(0)}'
        expr = BraKet(KetSymbol('Psi', hs=0), BasisKet(FockIndex(i), hs=0))
        assert latex(expr) == r'\Braket{\Psi | i}^(0)'
        expr = BraKet(BasisKet(FockIndex(i), hs=0), KetSymbol('Psi', hs=0))
        assert latex(expr) == r'\Braket{i | \Psi}^(0)'


def test_tex_bra_operations():
    """Test the tex representation of bra operations"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    psi1 = KetSymbol("Psi_1", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    psi2 = KetSymbol("Psi_2", hs=hs1)
    bra_psi1 = KetSymbol("Psi_1", hs=hs1).dag()
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag()
    bra_psi2 = KetSymbol("Psi_2", hs=hs1).dag()
    bra_psi3 = KetSymbol("Psi_3", hs=hs1).dag()
    bra_phi = KetSymbol("Phi", hs=hs2).dag()
    A = OperatorSymbol("A_0", hs=hs1)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    assert (
        latex((psi1 + psi2).dag()) ==
        r'\left\langle \Psi_{1} \right\rvert^{(q_{1})} + '
        r'\left\langle \Psi_{2} \right\rvert^{(q_{1})}')
    assert (
        latex((psi1 + psi2).dag(), tex_use_braket=True) ==
        r'\Bra{\Psi_{1}}^{(q_{1})} + \Bra{\Psi_{2}}^{(q_{1})}')
    assert (
        latex(bra_psi1 + bra_psi2) ==
        r'\left\langle \Psi_{1} \right\rvert^{(q_{1})} + '
        r'\left\langle \Psi_{2} \right\rvert^{(q_{1})}')
    assert (
        latex(bra_psi1 - bra_psi2 + bra_psi3) ==
        r'\left\langle \Psi_{1} \right\rvert^{(q_{1})} - '
        r'\left\langle \Psi_{2} \right\rvert^{(q_{1})} + '
        r'\left\langle \Psi_{3} \right\rvert^{(q_{1})}')
    assert (
        latex(bra_psi1 * bra_phi) ==
        r'\left\langle \Psi_{1} \right\rvert^{(q_{1})} \otimes '
        r'\left\langle \Phi \right\rvert^{(q_{2})}')
    assert (
        latex(bra_psi1 * bra_phi, tex_use_braket=True) ==
        r'\Bra{\Psi_{1}}^{(q_{1})} \otimes \Bra{\Phi}^{(q_{2})}')
    assert (
        latex(Bra(phase * psi1)) ==
        r'e^{i \gamma} \left\langle \Psi_{1} \right\rvert^{(q_{1})}')
    assert (
        latex((A * psi1).dag()) ==
        r'\left\langle \Psi_{1} \right\rvert^{(q_{1})} '
        r'\hat{A}_{0}^{(q_{1})\dagger}')


def test_tex_sop_elements():
    """Test the tex representation of "atomic" Superoperators"""
    hs1 = LocalSpace('q1', dimension=2)
    hs2 = LocalSpace('q2', dimension=2)
    alpha, beta = symbols('alpha, beta')
    assert latex(SuperOperatorSymbol("A", hs=hs1)) == r'\mathrm{A}^{(q_{1})}'
    assert (latex(SuperOperatorSymbol("A_1", hs=hs1*hs2)) ==
            r'\mathrm{A}_{1}^{(q_{1} \otimes q_{2})}')
    assert (latex(SuperOperatorSymbol("Xi", alpha, beta, hs=hs1)) ==
            r'\mathrm{\Xi}^{(q_{1})}\left(\alpha, \beta\right)')
    assert (latex(SuperOperatorSymbol("Xi_2", hs=('q1', 'q2'))) ==
            r'\mathrm{\Xi}_{2}^{(q_{1} \otimes q_{2})}')
    assert (latex(SuperOperatorSymbol("Xi_full", hs=1)) ==
            r'\mathrm{\Xi}_{\text{full}}^{(1)}')
    assert latex(IdentitySuperOperator) == r'\mathbb{1}'
    assert latex(ZeroSuperOperator) == r'\mathbb{0}'


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
    assert latex(A + B) == r'\mathrm{A}^{(q_{1})} + \mathrm{B}^{(q_{1})}'
    assert latex(A * B) == r'\mathrm{A}^{(q_{1})} \mathrm{B}^{(q_{1})}'
    assert latex(A * C) == r'\mathrm{A}^{(q_{1})} \mathrm{C}^{(q_{2})}'
    assert latex(2 * A) == r'2 \mathrm{A}^{(q_{1})}'
    assert latex(2j * A) == r'2i \mathrm{A}^{(q_{1})}'
    assert latex((1+2j) * A) == r'(1+2i) \mathrm{A}^{(q_{1})}'
    assert latex(gamma**2 * A) == r'\gamma^{2} \mathrm{A}^{(q_{1})}'
    assert (latex(-gamma**2/2 * A) ==
            r'- \frac{\gamma^{2}}{2} \mathrm{A}^{(q_{1})}')
    assert latex(SuperAdjoint(A)) == r'\mathrm{A}^{(q_{1})\dagger}'
    assert (latex(SuperAdjoint(A + B)) ==
            r'\left(\mathrm{A}^{(q_{1})} + '
            r'\mathrm{B}^{(q_{1})}\right)^\dagger')
    assert latex(A - B) == r'\mathrm{A}^{(q_{1})} - \mathrm{B}^{(q_{1})}'
    assert (latex(A - B + C) ==
            r'\mathrm{A}^{(q_{1})} - \mathrm{B}^{(q_{1})} + '
            r'\mathrm{C}^{(q_{2})}')
    assert (latex(2 * A - sqrt(gamma) * (B + C)) ==
            r'2 \mathrm{A}^{(q_{1})} - \sqrt{\gamma} '
            r'\left(\mathrm{B}^{(q_{1})} + \mathrm{C}^{(q_{2})}\right)')
    assert latex(SPre(A_op)) == r'\mathrm{SPre}\left(\hat{A}^{(1)}\right)'
    assert latex(SPost(A_op)) == r'\mathrm{SPost}\left(\hat{A}^{(1)}\right)'
    assert (latex(SuperOperatorTimesOperator(L, A_op)) ==
            r'\mathrm{L}^{(1)}\left[\hat{A}^{(1)}\right]')
    assert (latex(SuperOperatorTimesOperator(L, sqrt(gamma) * A_op)) ==
            r'\mathrm{L}^{(1)}\left[\sqrt{\gamma} \hat{A}^{(1)}\right]')
    assert (latex(SuperOperatorTimesOperator((L + 2*M), A_op)) ==
            r'\left(\mathrm{L}^{(1)} + 2 \mathrm{M}^{(1)}\right)'
            r'\left[\hat{A}^{(1)}\right]')


def test_tex_spin_arrows():
    """Test the representation of spin-1/2 spaces with special labels "down",
    "up" as arrows"""
    tls1 = SpinSpace('1', spin='1/2', basis=("down", "up"))
    tls2 = SpinSpace('2', spin='1/2', basis=("down", "up"))
    tls3 = SpinSpace('3', spin='1/2', basis=("down", "up"))
    down1 = BasisKet('down', hs=tls1)
    up1 = BasisKet('up', hs=tls1)
    down2 = BasisKet('down', hs=tls2)
    up3 = BasisKet('up', hs=tls3)
    assert latex(down1) == r'\left\lvert \downarrow \right\rangle^{(1)}'
    assert latex(up1) == r'\left\lvert \uparrow \right\rangle^{(1)}'
    ket = down1 * down2 * up3
    assert (
        latex(ket) ==
        r'\left\lvert \downarrow\downarrow\uparrow \right\rangle'
        r'^{(1 \otimes 2 \otimes 3)}')
    sig = LocalSigma("up", "down", hs=tls1)
    assert (
        latex(sig) ==
        r'\left\lvert \uparrow \middle\rangle\!'
        r'\middle\langle \downarrow \right\rvert^{(1)}')


@pytest.mark.xfail
def test_tex_spin_arrows_multi_sigma():
    # when fixed, combine with test_tex_spin_arrows
    tls1 = SpinSpace('1', spin='1/2', basis=("down", "up"))
    tls2 = SpinSpace('2', spin='1/2', basis=("down", "up"))
    tls3 = SpinSpace('3', spin='1/2', basis=("down", "up"))
    sig1 = LocalSigma("up", "down", hs=tls1)
    sig2 = LocalSigma("up", "up", hs=tls2)
    sig3 = LocalSigma("down", "down", hs=tls3)
    assert latex(sig1 * sig2 * sig3) == r''


def test_repr_latex():
    """Test the automatic representation in the notebook"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    assert A._repr_latex_() == "$%s$" % latex(A)
    assert (A + B)._repr_latex_() == "$%s$" % latex(A + B)


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

        def _latex(self, *args, **kwargs):
            return "%s(%s)" % (
                self._name, ", ".join(
                    [latex(sym) for sym in self._sym_args]))

    return ScalarFunc


def test_tex_derivative(MyScalarFunc):
    s, s0, t, t0, gamma = symbols('s, s_0, t, t_0, gamma', real=True)
    m = IdxSym('m')
    n = IdxSym('n')
    S = IndexedBase('s')
    T = IndexedBase('t')

    f = partial(MyScalarFunc, "f")
    g = partial(MyScalarFunc, "g")

    expr = f(s, t).diff(t)
    assert latex(expr) == r'\frac{\partial}{\partial t} f(s, t)'

    expr = f(s, t).diff(s, n=2).diff(t)
    assert latex(expr) == (
        r'\frac{\partial^{3}}{\partial s^{2} \partial t} f(s, t)')

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: s0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial s^{2} \partial t} f(s, t) '
        r'\right\vert_{s=s_{0}}')

    expr = f(S[m], T[n]).diff(S[m], n=2).diff(T[n]).evaluate_at({S[m]: s0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial s_{m}^{2} \partial t_{n}} '
        r'f(s_{m}, t_{n}) \right\vert_{s_{m}=s_{0}}')

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: 0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial s^{2} \partial t} f(s, t) '
        r'\right\vert_{s=0}')

    expr = f(gamma, t).diff(gamma, n=2).diff(t).evaluate_at({gamma: 0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial \gamma^{2} \partial t} '
        r'f(\gamma, t) \right\vert_{\gamma=0}')

    expr = f(s, t).diff(s, n=2).diff(t).evaluate_at({s: s0, t: t0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial s^{2} \partial t} f(s, t) '
        r'\right\vert_{s=s_{0}, t=t_{0}}')

    D = expr.__class__

    expr = D(f(s, t) + g(s, t), derivs={s: 2, t: 1}, vals={s: s0, t: t0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial s^{2} \partial t} '
        r'\left(f(s, t) + g(s, t)\right) \right\vert_{s=s_{0}, t=t_{0}}')

    expr = D(2 * f(s, t), derivs={s: 2, t: 1}, vals={s: s0, t: t0})
    assert latex(expr) == (
        r'\left. \frac{\partial^{3}}{\partial s^{2} \partial t} '
        r'\left(2 f(s, t)\right) \right\vert_{s=s_{0}, t=t_{0}}')

    expr = f(s, t).diff(t) * g(s, t)
    assert latex(expr) == (
        r'\left(\frac{\partial}{\partial t} f(s, t)\right) g(s, t)')

    expr = f(s, t).diff(t).evaluate_at({t: 0}) * g(s, t)
    assert latex(expr) == (
        r'\left(\left. \frac{\partial}{\partial t} f(s, t) '
        r'\right\vert_{t=0}\right) g(s, t)')

    expr = f(s, t).diff(t) + g(s, t)
    assert latex(expr) == r'\frac{\partial}{\partial t} f(s, t) + g(s, t)'

    f = MyScalarFunc("f", S[m], T[n])
    series = f.series_expand(T[n], about=0, order=3)
    assert latex(series) == (
        r'\left(f(s_{m}, 0), \left. \frac{\partial}{\partial t_{n}} '
        r'f(s_{m}, t_{n}) \right\vert_{t_{n}=0}, \frac{1}{2} \left(\left. '
        r'\frac{\partial^{2}}{\partial t_{n}^{2}} f(s_{m}, t_{n}) '
        r'\right\vert_{t_{n}=0}\right), \frac{1}{6} \left(\left. '
        r'\frac{\partial^{3}}{\partial t_{n}^{3}} f(s_{m}, t_{n}) '
        r'\right\vert_{t_{n}=0}\right)\right)')
    f = MyScalarFunc("f", s, t)
    series = f.series_expand(t, about=0, order=2)
    assert (
        latex(series) ==
        r'\left(f(s, 0), \left. \frac{\partial}{\partial t} f(s, t) '
        r'\right\vert_{t=0}, \frac{1}{2} \left(\left. '
        r'\frac{\partial^{2}}{\partial t^{2}} f(s, t) '
        r'\right\vert_{t=0}\right)\right)')

    expr = (  # nested derivative
        MyScalarFunc("f", s, t)
        .diff(s, n=2)
        .diff(t)
        .evaluate_at({t: t0})
        .diff(t0))
    assert latex(expr) == (
        r'\frac{\partial}{\partial t_{0}} \left(\left. '
        r'\frac{\partial^{3}}{\partial s^{2} \partial t} f(s, t) '
        r'\right\vert_{t=t_{0}}\right)')
