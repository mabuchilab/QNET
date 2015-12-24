import sympy
from qnet.misc.qsd_codegen import (local_ops, find_kets, QSDCodeGen,
    QSDOperator, QSDCodeGenError, UNSIGNED_MAXINT, expand_cmd,
    compilation_worker, qsd_run_worker)
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, SLH, set_union, TrivialSpace, symbols, sqrt,
    LocalSigma, identity_matrix, I
)
from qnet.algebra.state_algebra import (
    BasisKet, LocalKet, TensorKet, CoherentStateKet
)
from qnet.algebra.hilbert_space_algebra import BasisRegistry
import os
import shutil
import stat
import re
from textwrap import dedent
from collections import OrderedDict
from qnet.circuit_components.pseudo_nand_cc import PseudoNAND
from qnet.misc.testing_tools import datadir, qsd_traj, fake_traj
import numpy as np
import pytest
try:
    import unittest.mock as mock
except ImportError:
    import mock
# built-in fixtures: tmpdir, request, monkeypatch
# pytest-capturelog fixtures: caplog


datadir = pytest.fixture(datadir)
TRAJ1_SEED = 103212
traj1    = pytest.fixture(qsd_traj(datadir, 'traj1', TRAJ1_SEED))
TRAJ2_SEED = 18322321
traj2_10 = pytest.fixture(qsd_traj(datadir, 'traj2_10', TRAJ2_SEED))


def test_local_ops():
    psa = PseudoNAND()
    assert isinstance(psa, Circuit)
    l_ops = local_ops(psa)
    a = Destroy(psa.space)
    assert type(local_ops(a)) is set
    assert set([IdentityOperator, a, a.dag()]) == l_ops
    assert local_ops(a) == set([a])
    assert local_ops(a*a) == set([a])
    assert local_ops(a + a.dag()) == set([a,a.dag()])
    assert local_ops(10*a) == set([a])
    with pytest.raises(TypeError):
        local_ops({})


def test_find_kets():
    #                  hs n
    psi_A_0 = BasisKet(0, 0)
    psi_A_1 = BasisKet(0, 1)
    psi_B_0 = BasisKet(1, 0)
    psi_B_1 = BasisKet(1, 1)
    local_psi = [psi_A_0, psi_A_1, psi_B_0, psi_B_1]

    psi_00 = psi_A_0 * psi_B_0
    psi_01 = psi_A_0 * psi_B_1
    psi_10 = psi_A_1 * psi_B_0
    psi_11 = psi_A_1 * psi_B_1
    tensor_psi = [psi_00, psi_01, psi_10, psi_11]
    a1 = Destroy(psi_A_1.space)

    assert set((psi_A_0, )) == find_kets(psi_A_0, LocalKet)

    psi = 0.5*(psi_00 + psi_01 + psi_10 + psi_11)
    assert set(local_psi) == find_kets(psi, LocalKet)
    assert set(tensor_psi) == find_kets(psi, TensorKet)

    psi = 0.5 * a1 * psi_10 # = 0.5 * psi_00
    assert set([psi_A_0, psi_B_0]) == find_kets(psi, LocalKet)
    assert set([psi_00, ]) == find_kets(psi, TensorKet)

    with pytest.raises(TypeError):
        find_kets({}, cls=LocalKet)


def test_qsd_codegen_operator_basis():
    a = Destroy(1)
    ad = a.dag()
    s = LocalSigma(2, 1, 0)
    sd = s.dag()
    circuit = SLH(identity_matrix(0), [], a*ad + s + sd)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    IdentityOperator Id0(0);
    IdentityOperator Id1(1);
    AnnihilationOperator A0(0);
    FieldTransitionOperator S1_0_1(0,1,1);
    FieldTransitionOperator S1_1_0(1,0,1);
    Operator Id = Id0*Id1;
    Operator Ad0 = A0.hc();
    """).strip()
    circuit = SLH(identity_matrix(0), [], ad)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    IdentityOperator Id0(0);
    AnnihilationOperator A0(0);
    Operator Id = Id0;
    Operator Ad0 = A0.hc();
    """).strip()


def test_qsd_codegen_parameters():
    k = symbols(r'kappa', positive=True)
    x = symbols(r'chi', real=True)
    c = symbols("c")

    a = Destroy(1)
    H = x * (a * a + a.dag() * a.dag()) + (c * a.dag() + c.conjugate() * a)
    L = sqrt(k) * a
    slh = SLH(identity_matrix(1), [L], H)
    codegen = QSDCodeGen(circuit=slh, num_vals={x: 2., c: 1+2j, k: 2})

    scode = codegen._parameters_lines()
    assert dedent(scode).strip() == dedent("""
    Complex I(0.0,1.0);
    Complex c(1,2);
    double chi = 2;
    double kappa = 2;""").strip()

    codegen.num_vals.update({c: 1})
    scode = codegen._parameters_lines()
    assert dedent(scode).strip() == dedent("""
    Complex I(0.0,1.0);
    Complex c(1,0);
    double chi = 2;
    double kappa = 2;""").strip()

    del codegen.num_vals[c]
    with pytest.raises(KeyError) as excinfo:
        scode = codegen._parameters_lines()
    assert "There is no value for symbol c" in str(excinfo.value)


def test_latex_symbols(slh_Sec6):
    """Test that if any of the symbols contain "special" characters such as
    backslashes (LaTeX code), we still get valid C++ code (basic ASCII words
    only)."""
    k = symbols("\kappa", positive=True)
    x = symbols(r'\chi^{(1)}_{\text{main}}', real=True)
    c = symbols("c")

    a = Destroy(1)
    H = x * (a * a + a.dag() * a.dag()) + (c * a.dag() + c.conjugate() * a)
    L = sqrt(k) * a
    slh = SLH(identity_matrix(1), [L], H)
    codegen = QSDCodeGen(circuit=slh, num_vals={x: 2., c: 1+2j, k: 2})

    scode = codegen._parameters_lines()
    assert dedent(scode).strip() == dedent("""
    Complex I(0.0,1.0);
    Complex c(1,2);
    double chi_1_textmain = 2;
    double kappa = 2;""").strip()


def test_qsd_codegen_hamiltonian():
    k = symbols(r'\kappa', positive=True)
    x = symbols(r'\chi', real=True)
    c = symbols("c",real=True)

    a = Destroy(1)
    H = x * (a * a + a.dag() * a.dag()) + (c * a.dag() + c.conjugate() * a)
    L = sqrt(k) * a
    slh = SLH(identity_matrix(1), [L], H)
    codegen = QSDCodeGen(circuit=slh, num_vals={x: 2., c: 1, k: 2})

    codegen._operator_basis_lines()
    scode = codegen._hamiltonian_lines()
    assert scode.strip() == (r'Operator H = ((c) * (Ad0) + (c) * (A0) + (chi) '
                             r'* (((Ad0 * Ad0) + (A0 * A0))));')

def test_QSDOperator():
    A0 = QSDOperator('AnnihilationOperator', 'A0', '(0)')
    assert A0.__repr__() == "QSDOperator('AnnihilationOperator', 'A0', '(0)')"
    assert A0.qsd_type == 'AnnihilationOperator'
    assert A0.name == 'A0'
    assert A0.instantiator == '(0)'
    assert A0.instantiation == 'AnnihilationOperator A0(0);'
    assert str(A0) == str(A0.name)
    assert len(A0) == 3
    qsd_type, name, instantiator = A0
    assert qsd_type == A0.qsd_type
    assert name == A0.name
    assert instantiator == A0.instantiator

    Id0 = QSDOperator('IdentityOperator', 'Id0', '(0)')
    assert Id0.instantiation == 'IdentityOperator Id0(0);'

    S = QSDOperator('FieldTransitionOperator', 'S1_0_1', '(0,1,1)')
    assert S.instantiation == 'FieldTransitionOperator S1_0_1(0,1,1);'

    Ad0 = QSDOperator('Operator', 'Ad0', '= A0.hc()')
    assert Ad0.instantiator == '= A0.hc()'
    assert Ad0.instantiation == 'Operator Ad0 = A0.hc();'

    # spaces around the '=' should be stripped out
    Ad0_2 = QSDOperator('Operator', 'Ad0', '  =   A0.hc()')
    assert Ad0_2 == Ad0
    Ad0_2 = QSDOperator('Operator', 'Ad0', '=A0.hc()')
    assert Ad0_2 == Ad0

    # having __hash__ implemente allows usage in dicts and sets
    d = {Ad0: 1}
    assert d[Ad0] == 1
    assert d[Ad0_2] == 1
    s = set([Ad0, ])
    assert Ad0_2 in s

    # Changing an attribute is a change in __key()
    Ad0.name = 'Adagger0'
    assert Ad0 != Ad0_2
    with pytest.raises(KeyError):
        v = d[Ad0]
    assert Ad0 not in s
    d[Ad0] = 2
    assert len(d) == 2

    with pytest.raises(ValueError) as excinfo:
        Ad0 = QSDOperator('CreationOperator', 'Ad0', '(0)')
    assert "Type 'CreationOperator' must be one of" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        A0 = QSDOperator('AnnihilationOperator', 'A 0', '(0)')
    assert "Name 'A 0' is not a valid C++ variable name" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Ad0 = QSDOperator('Operator', 'Ad0', 'A0.hc()')
    assert "Instantiator 'A0.hc()' does not start with" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Id0 = QSDOperator('IdentityOperator', 'Id0', '(0')
    assert "Instantiator '(0' does not end with ')'" in str(excinfo.value)


@pytest.fixture
def slh_Sec6():
    """SHL for the model in Section 6 of the QSD paper"""
    E      = symbols(r'E',        positive=True)
    chi    = symbols(r'\chi',     real=True)
    omega  = symbols(r'\omega',   real=True)
    eta    = symbols(r'\eta',     real=True)
    gamma1 = symbols(r'\gamma_1', positive=True)
    gamma2 = symbols(r'\gamma_2', positive=True)
    kappa  = symbols(r'\kappa',   positive=True)
    A1 = Destroy(0)
    Ac1 = A1.dag()
    N1 = Ac1*A1
    Id1 = identity_matrix(0)
    A2 = Destroy(1)
    Ac2 = A2.dag()
    N2 = Ac2*A2
    Id2 = identity_matrix(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    Id3 = identity_matrix(3)

    H  = E*I*(Ac1-A1) + 0.5*chi*I*(Ac1*Ac1*A2 - A1*A1*Ac2) \
         + omega*Sp*Sm + eta*I*(A2*Sp-Ac2*Sm)
    Lindblads = [sqrt(2*gamma1)*A1, sqrt(2*gamma2)*A2, sqrt(2*kappa)*Sm]

    return SLH(identity_matrix(3), Lindblads, H)


@pytest.fixture
def slh_Sec6_vals():
    return {
        symbols(r'E',        positive=True):    20.0,
        symbols(r'\chi',     real=True):         0.4,
        symbols(r'\omega',   real=True):        -0.7,
        symbols(r'\eta',     real=True):         0.001,
        symbols(r'\gamma_1', positive=True):     1.0,
        symbols(r'\gamma_2', positive=True):     1.0,
        symbols(r'\kappa',   positive=True):     0.1
    }


@pytest.fixture
def Sec6_codegen(slh_Sec6, slh_Sec6_vals):
    codegen = QSDCodeGen(circuit=slh_Sec6, num_vals=slh_Sec6_vals)
    A2 = Destroy(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    codegen.add_observable(Sp*A2*Sm*Sp, name="X1")
    codegen.add_observable(Sm*Sp*A2*Sm, name="X2")
    codegen.add_observable(A2, name="A2")
    psi0 = BasisKet(0, 0)
    psi1 = BasisKet(1, 0)
    psi2 = BasisKet(2, 0)
    BasisRegistry.set_basis(psi0.space, range(50))
    BasisRegistry.set_basis(psi1.space, range(50))
    BasisRegistry.set_basis(psi2.space, range(2))
    codegen.set_trajectories(psi_initial=psi0*psi1*psi2,
            stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            traj_save=10)
    return codegen


def test_operator_str(Sec6_codegen):
    gamma1 = symbols(r'\gamma_1', positive=True)
    A0 = Destroy(0)
    Op = sqrt(gamma1)*A0
    assert Sec6_codegen._operator_str(Op) == '(sqrt(gamma_1)) * (A0)'


def test_qsd_codegen_lindblads(slh_Sec6):
    codegen = QSDCodeGen(circuit=slh_Sec6)
    scode = codegen._lindblads_lines()
    assert dedent(scode).strip() == dedent(r'''
    const int nL = 3;
    Operator L[nL]={
      (sqrt(2)*sqrt(gamma_1)) * (A0),
      (sqrt(2)*sqrt(gamma_2)) * (A1),
      (sqrt(2)*sqrt(kappa)) * (S2_0_1)
    };
    ''').strip()


def test_qsd_codegen_observables(caplog, slh_Sec6, slh_Sec6_vals):
    A2 = Destroy(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    codegen = QSDCodeGen(circuit=slh_Sec6, num_vals=slh_Sec6_vals)

    with pytest.raises(QSDCodeGenError) as excinfo:
        scode = codegen._observables_lines()
    assert "Must register at least one observable" in str(excinfo.value)

    codegen.add_observable(Sp*A2*Sm*Sp)
    name = 'a_1 sigma_10^[2]'
    filename = codegen._observables[name][1]
    assert filename == 'a_1_sigma_10_2.out'
    codegen.add_observable(Sp*A2*Sm*Sp)
    assert 'Overwriting existing operator' in caplog.text()

    with pytest.raises(ValueError) as exc_info:
        codegen.add_observable(Sp*A2*A2*Sm*Sp)
    assert "longer than limit" in str(exc_info.value)
    name = 'A2^2'
    codegen.add_observable(Sp*A2*A2*Sm*Sp, name=name)
    assert name in codegen._observables
    filename = codegen._observables[name][1]
    assert filename == 'A2_2.out'

    with pytest.raises(ValueError) as exc_info:
        codegen.add_observable(A2, name='A2_2')
    assert "Cannot generate unique filename" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        codegen.add_observable(A2, name="A2\t2")
    assert "invalid characters" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        codegen.add_observable(A2, name="A"*100)
    assert "longer than limit" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        codegen.add_observable(A2, name="()")
    assert "Cannot generate filename" in str(exc_info.value)

    codegen = QSDCodeGen(circuit=slh_Sec6, num_vals=slh_Sec6_vals)
    codegen.add_observable(Sp*A2*Sm*Sp, name="X1")
    codegen.add_observable(Sm*Sp*A2*Sm, name="X2")
    assert codegen._observables["X2"] == (Sm*Sp*A2*Sm, 'X2.out')
    codegen.add_observable(A2, name="A2")
    assert codegen._observables["A2"] == (A2, 'A2.out')
    scode = codegen._observables_lines()
    assert dedent(scode).strip() == dedent(r'''
    const int nOfOut = 3;
    Operator outlist[nOfOut] = {
      (A1 * S2_1_0),
      (A1 * S2_0_1),
      A1
    };
    char *flist[nOfOut] = {"X1.out", "X2.out", "A2.out"};
    int pipe[4] = {1,2,3,4};
    ''').strip()
    # Note how the observables have been simplified
    assert Sp*A2*Sm*Sp == Sp*A2
    assert codegen._operator_str(Sp*A2) == '(A1 * S2_1_0)'
    assert Sm*Sp*A2*Sm == Sm*A2
    assert codegen._operator_str(Sm*A2) == '(A1 * S2_0_1)'
    # If the oberservables introduce new operators or symbols, these should
    # extend the existing ones
    P1 = LocalSigma(2, 1, 1)
    zeta = symbols("zeta", real=True)
    codegen.add_observable(zeta*P1, name="P1")
    assert P1 in codegen._local_ops
    assert str(codegen._qsd_ops[P1]) == 'S2_1_1'
    assert zeta in codegen.syms
    codegen.num_vals.update({zeta: 1.0})
    assert 'zeta' in codegen._parameters_lines()
    assert str(codegen._qsd_ops[P1]) in codegen._operator_basis_lines()
    assert Sp*A2 in set(codegen.observables)
    assert Sm*A2 in set(codegen.observables)
    assert zeta*P1 in set(codegen.observables)
    assert list(codegen.observable_names) == ['X1', 'X2', 'A2', 'P1']
    assert codegen.get_observable('X1') == Sp*A2*Sm*Sp


def test_ordered_tensor_operands(slh_Sec6):
    codegen = QSDCodeGen(circuit=slh_Sec6)
    psi = BasisKet(0, 0) * BasisKet(1, 0)
    assert (list(psi.operands) == list(codegen._ordered_tensor_operands(psi)))
    psi = TensorKet(BasisKet(1, 0), BasisKet(0, 0))
    assert (list(reversed(psi.operands))
            == list(codegen._ordered_tensor_operands(psi)))


def test_define_atomic_kets(slh_Sec6):
    codegen = QSDCodeGen(circuit=slh_Sec6)
    psi_cav1 = lambda n:  BasisKet(0, n)
    psi_cav2 = lambda n:  BasisKet(1, n)
    psi_spin = lambda n:  BasisKet(2, n)
    psi_tot = lambda n, m, l: psi_cav1(n) * psi_cav2(m) * psi_spin(l)

    with pytest.raises(QSDCodeGenError) as excinfo:
        lines = codegen._define_atomic_kets(psi_cav1(0))
    assert "not in the Hilbert space of the Hamiltonian" in str(excinfo.value)
    BasisRegistry.registry = {} # reset
    with pytest.raises(QSDCodeGenError) as excinfo:
        lines = codegen._define_atomic_kets(psi_tot(0,0,0))
    assert "Unknown dimension for Hilbert space" in str(excinfo.value)

    psi_cav1(0).space.dimension = 10
    psi_cav2(0).space.dimension = 10
    psi_spin(0).space.dimension = 2

    lines = codegen._define_atomic_kets(psi_tot(0,0,0))
    scode = "\n".join(lines)
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 0
    State phiL1(10,0,FIELD); // HS 1
    State phiL2(2,0,FIELD); // HS 2
    State phiT0List[3] = {phiL0, phiL1, phiL2};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2
    ''').strip()

    psi = ( ((psi_cav1(0) + psi_cav1(1)) / sympy.sqrt(2))
          * ((psi_cav2(0) + psi_cav2(1)) / sympy.sqrt(2))
          * psi_spin(0) )
    lines = codegen._define_atomic_kets(psi)
    scode = "\n".join(lines)
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 0
    State phiL1(10,0,FIELD); // HS 1
    State phiL2(2,0,FIELD); // HS 2
    State phiL3(10,1,FIELD); // HS 0
    State phiL4(10,1,FIELD); // HS 1
    State phiT0List[3] = {(phiL0 + phiL3), (phiL1 + phiL4), phiL2};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2
    ''').strip()
    for phi in (
          list(find_kets(psi, cls=LocalKet))
        + list(find_kets(psi, cls=TensorKet))
    ):
        assert phi in codegen._qsd_states

    alpha = symbols('alpha')
    psi = CoherentStateKet(0, alpha) * psi_cav2(0) * psi_spin(0)
    with pytest.raises(TypeError) as excinfo:
        lines = codegen._define_atomic_kets(psi)
    assert "neither a known symbol nor a complex number" in str(excinfo.value)
    codegen.syms.add(alpha)
    codegen._update_var_names()
    lines = codegen._define_atomic_kets(psi)
    scode = "\n".join(lines)
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 1
    State phiL1(2,0,FIELD); // HS 2
    State phiL2(10,alpha,FIELD); // HS 0
    State phiT0List[3] = {phiL2, phiL0, phiL1};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2
    ''').strip()
    for phi in (
          list(find_kets(psi, cls=LocalKet))
        + list(find_kets(psi, cls=TensorKet))
    ):
        assert phi in codegen._qsd_states

    psi = CoherentStateKet(0, 1j) * psi_cav2(0) * psi_spin(0)
    lines = codegen._define_atomic_kets(psi)
    scode = "\n".join(lines)
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 1
    State phiL1(2,0,FIELD); // HS 2
    Complex phiL2_alpha(0,1);
    State phiL2(10,phiL2_alpha,FIELD); // HS 0
    State phiT0List[3] = {phiL2, phiL0, phiL1};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2
    ''').strip()
    for phi in (
          list(find_kets(psi, cls=LocalKet))
        + list(find_kets(psi, cls=TensorKet))
    ):
        assert phi in codegen._qsd_states

    psi = psi_tot(1,0,0) + psi_tot(0,1,0) + psi_tot(0,0,1)
    lines = codegen._define_atomic_kets(psi)
    scode = "\n".join(lines)
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 0
    State phiL1(10,0,FIELD); // HS 1
    State phiL2(2,0,FIELD); // HS 2
    State phiL3(10,1,FIELD); // HS 0
    State phiL4(10,1,FIELD); // HS 1
    State phiL5(2,1,FIELD); // HS 2
    State phiT0List[3] = {phiL0, phiL1, phiL5};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2
    State phiT1List[3] = {phiL0, phiL4, phiL2};
    State phiT1(3, phiT1List); // HS 0 * HS 1 * HS 2
    State phiT2List[3] = {phiL3, phiL1, phiL2};
    State phiT2(3, phiT2List); // HS 0 * HS 1 * HS 2
    ''').strip()
    for phi in (
          list(find_kets(psi, cls=LocalKet))
        + list(find_kets(psi, cls=TensorKet))
    ):
        assert phi in codegen._qsd_states


def test_qsd_codegen_initial_state(slh_Sec6):

    A2 = Destroy(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    psi_cav1 = lambda n:  BasisKet(0, n)
    psi_cav2 = lambda n:  BasisKet(1, n)
    psi_spin = lambda n:  BasisKet(2, n)
    psi_tot = lambda n, m, l: psi_cav1(n) * psi_cav2(m) * psi_spin(l)

    psi_cav1(0).space.dimension = 10
    psi_cav2(0).space.dimension = 10
    psi_spin(0).space.dimension = 2

    codegen = QSDCodeGen(circuit=slh_Sec6)
    codegen.add_observable(Sp*A2*Sm*Sp, "X1.out")
    codegen.add_observable(Sm*Sp*A2*Sm, "X2.out")
    codegen.add_observable(A2, "A2.out")

    psi = ( ((psi_cav1(0) + psi_cav1(1)) / sympy.sqrt(2))
          * ((psi_cav2(0) + psi_cav2(1)) / sympy.sqrt(2))
          * psi_spin(0) )
    codegen.set_trajectories(psi_initial=psi, stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            traj_save=10)

    scode = codegen._initial_state_lines()
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 0
    State phiL1(10,0,FIELD); // HS 1
    State phiL2(2,0,FIELD); // HS 2
    State phiL3(10,1,FIELD); // HS 0
    State phiL4(10,1,FIELD); // HS 1
    State phiT0List[3] = {(phiL0 + phiL3), (phiL1 + phiL4), phiL2};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2

    State psiIni = (1/2) * (phiT0);
    psiIni.normalize();
    ''').strip()
    # TODO: Fix 1/2 => 1.0/2.0

    alpha = symbols('alpha')
    psi = CoherentStateKet(0, alpha) * psi_cav2(0) * psi_spin(0)
    codegen.set_trajectories(psi_initial=psi, stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            traj_save=10)
    scode = codegen._initial_state_lines()
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 1
    State phiL1(2,0,FIELD); // HS 2
    State phiL2(10,alpha,FIELD); // HS 0
    State phiT0List[3] = {phiL2, phiL0, phiL1};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2

    State psiIni = phiT0;
    psiIni.normalize();
    ''').strip()

    psi = (psi_tot(1,0,0) + psi_tot(0,1,0)) / sympy.sqrt(2)
    codegen.set_trajectories(psi_initial=psi, stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            traj_save=10)
    scode = codegen._initial_state_lines()
    assert scode == dedent(r'''
    State phiL0(10,0,FIELD); // HS 0
    State phiL1(10,0,FIELD); // HS 1
    State phiL2(2,0,FIELD); // HS 2
    State phiL3(10,1,FIELD); // HS 0
    State phiL4(10,1,FIELD); // HS 1
    State phiT0List[3] = {phiL0, phiL4, phiL2};
    State phiT0(3, phiT0List); // HS 0 * HS 1 * HS 2
    State phiT1List[3] = {phiL3, phiL1, phiL2};
    State phiT1(3, phiT1List); // HS 0 * HS 1 * HS 2

    State psiIni = (sqrt(2)/2) * ((phiT0 + phiT1));
    psiIni.normalize();
    ''').strip()


def test_qsd_codegen_traj(slh_Sec6):
    A2 = Destroy(1)
    Sp = LocalSigma(2, 1, 0)
    Sm = Sp.dag()
    codegen = QSDCodeGen(circuit=slh_Sec6)
    codegen.add_observable(Sp*A2*Sm*Sp, name="X1")
    codegen.add_observable(Sm*Sp*A2*Sm, name="X2")
    codegen.add_observable(A2, name="A2")

    with pytest.raises(QSDCodeGenError) as excinfo:
        scode = codegen._trajectory_lines()
    assert "No trajectories set up"  in str(excinfo.value)

    codegen.set_trajectories(psi_initial=None, stepper='AdaptiveStep', dt=0.01,
            nt_plot_step=100, n_plot_steps=5, n_trajectories=1,
            traj_save=10)
    scode = codegen._trajectory_lines()
    assert dedent(scode).strip() == dedent(r'''
    ACG gen(rndSeed); // random number generator
    ComplexNormal rndm(&gen); // Complex Gaussian random numbers

    double dt = 0.01;
    int dtsperStep = 100;
    int nOfSteps = 5;
    int nTrajSave = 10;
    int nTrajectory = 1;
    int ReadFile = 0;

    AdaptiveStep stepper(psiIni, H, nL, L);
    Trajectory traj(psiIni, dt, stepper, &rndm);

    traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,
                nTrajectory, nTrajSave, ReadFile);
    ''').strip()

    with pytest.raises(ValueError) as excinfo:
        codegen.set_moving_basis(move_dofs=0, delta=0.01, width=2,
                                 move_eps=0.01)
    assert "move_dofs must be an integer >0" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        codegen.set_moving_basis(move_dofs=4, delta=0.01, width=2,
                                 move_eps=0.01)
    assert "move_dofs must not be larger" in str(excinfo.value)
    with pytest.raises(QSDCodeGenError) as excinfo:
        codegen.set_moving_basis(move_dofs=3, delta=0.01, width=2,
                                 move_eps=0.01)
    assert "A moving basis cannot be used" in str(excinfo.value)
    codegen.set_moving_basis(move_dofs=2, delta=0.01, width=2, move_eps=0.01)
    scode = codegen._trajectory_lines()
    assert dedent(scode).strip() == dedent(r'''
    ACG gen(rndSeed); // random number generator
    ComplexNormal rndm(&gen); // Complex Gaussian random numbers

    double dt = 0.01;
    int dtsperStep = 100;
    int nOfSteps = 5;
    int nTrajSave = 10;
    int nTrajectory = 1;
    int ReadFile = 0;

    AdaptiveStep stepper(psiIni, H, nL, L);
    Trajectory traj(psiIni, dt, stepper, &rndm);

    int move = 2;
    double delta = 0.01;
    int width = 2;
    double moveEps = 0.01;

    traj.sumExp(nOfOut, outlist, flist , dtsperStep, nOfSteps,
                nTrajectory, nTrajSave, ReadFile, move,
                delta, width, moveEps);
    ''').strip()


def test_generate_code(datadir, Sec6_codegen):
    scode = Sec6_codegen.generate_code()
    #with(open("scode.out", 'w')) as out_fh:
        #out_fh.write(scode)
        #out_fh.write("\n")
    with open(os.path.join(datadir, 'Sec6.cc')) as in_fh:
        scode_expected = in_fh.read()
    assert scode.strip()  == scode_expected.strip()


def test_expand_cmd(monkeypatch):
    monkeypatch.setenv('HOME', '/home/qnet')
    monkeypatch.setenv('PREFIX', '/home/qnet/local')
    with pytest.raises(TypeError) as exc_info:
        expand_cmd("$CC -O2 -I$PREFIX/include/qsd -o qsd-example $HOME/qsd-example.cc -L$PREFIX/lib -lqsd")
    assert "cmd must be a list" in str(exc_info.value)
    cmd = ["$CC", "-O2", "-I$PREFIX/include/qsd", "-o qsd-example",
           "$HOME/qsd-example.cc", "-L$PREFIX/lib", "-lqsd"]
    assert " ".join(expand_cmd(cmd)) == '$CC -O2 -I/home/qnet/local/include/qsd -o qsd-example /home/qnet/qsd-example.cc -L/home/qnet/local/lib -lqsd'


def test_compilation_worker(tmpdir, monkeypatch, Sec6_codegen):
    HOME = str(tmpdir)
    PREFIX = os.path.join(HOME, 'local')
    monkeypatch.setenv('HOME', HOME)
    monkeypatch.setenv('PREFIX', PREFIX)
    scode = Sec6_codegen.generate_code()

    cmd = ["CC", "-O2", "-I$PREFIX/include/qsd", "-o run_qsd1",
           "run_qsd1.cc", "-L$PREFIX/lib", "-lqsd"]
    kwargs = {'executable': 'run_qsd1', 'path': '~/tmp', 'cc_code': scode,
            'keep_cc': True, 'cmd': cmd}
    def runner1(cmd, stderr, cwd):
        """simulate invocation of the compiler"""
        assert " ".join(expand_cmd(cmd)) == 'CC -O2 -I'+PREFIX+'/include/qsd -o run_qsd1 run_qsd1.cc -L'+PREFIX+'/lib -lqsd'
        assert cwd == os.path.join(HOME, 'tmp')
        executable = os.path.join(HOME, 'tmp', 'run_qsd1')
        with open(executable, 'w') as out_fh:
            out_fh.write("#!/bin/bash\n")
            out_fh.write("echo 'Hello World'\n")
        st = os.stat(executable)
        os.chmod(executable, st.st_mode | stat.S_IEXEC)
    executable = compilation_worker(kwargs, _runner=runner1)
    assert os.path.isfile(os.path.join(HOME, 'tmp', 'run_qsd1.cc'))
    assert executable == os.path.join(HOME, 'tmp', 'run_qsd1')
    assert os.path.isfile(executable)


def test_qsd_run_worker(datadir, tmpdir, monkeypatch):
    # set up an isolated environment
    HOME = str(tmpdir)
    PREFIX = os.path.join(HOME, 'local')
    monkeypatch.setenv('HOME', HOME)
    monkeypatch.setenv('PREFIX', PREFIX)
    os.mkdir(os.path.join(HOME, 'jobs'))
    os.mkdir(os.path.join(HOME, 'bin'))
    executable = os.path.join(HOME, 'bin', 'run_qsd')
    workdir = os.path.join(HOME, 'jobs', 'traj1')
    with open(executable, 'w') as out_fh:
        out_fh.write("#!/bin/bash\n")
        out_fh.write("echo 'Hello World'\n")
    st = os.stat(executable)
    os.chmod(executable, st.st_mode | stat.S_IEXEC)
    assert not os.path.isdir(workdir)
    # mock runner (needs to create the expected output files in workdir)
    def runner(cmd, stderr, cwd):
        """simulate invocation of the of the compiled program"""
        assert cwd == workdir
        assert os.path.isdir(cwd) # should have been created
        assert " ".join(expand_cmd(cmd)) == HOME+'/bin/run_qsd 232334'
        shutil.copy(os.path.join(datadir, 'X1.out'), workdir)
        shutil.copy(os.path.join(datadir, 'X2.out'), workdir)
    # run the worker
    kwargs = {'executable': 'run_qsd', 'path': '~/bin',
            'seed': 232334, 'operators': {'X1': 'X1.out', 'X2': 'X2.out'},
            'workdir': '~/jobs/traj1', 'keep': False}
    traj = qsd_run_worker(kwargs, _runner=runner)
    assert not os.path.isdir(workdir)
    assert os.path.isdir(os.path.join(HOME, 'jobs'))
    kwargs = {'executable': 'run_qsd', 'path': '~/bin',
            'seed': 232334, 'operators': {'X1': 'X1.out', 'X2': 'X2.out'},
            'workdir': '~/jobs/traj1', 'keep': True}
    traj = qsd_run_worker(kwargs, _runner=runner)
    assert traj.ID == 'd9831647-f2e7-3793-8b24-7c49c5c101a7'
    assert os.path.isfile(os.path.join(workdir, 'X1.out'))


def test_compile(Sec6_codegen):
    traj = Sec6_codegen
    assert traj.compile_cmd == ''
    with pytest.raises(ValueError) as exc_info:
        traj.compile(qsd_lib='~/local/lib', qsd_headers='~/local/header',
                    executable='qsd_test', path='~/bin', compiler='$CC',
                    compile_options='-g -O0', delay=True, keep_cc=False)
    assert "point to a file of the name libqsd.a" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        traj.compile(qsd_lib='~/local/lib/libqsd.a',
                    qsd_headers='~/local/header', executable='~/bin/qsd_test',
                    path='~/bin', compiler='$CC', compile_options='-g -O0',
                    delay=True, keep_cc=False)
    assert "Invalid executable name" in str(exc_info.value)
    traj.compile(qsd_lib='~/local/lib/libqsd.a',
                qsd_headers='~/local/header', executable='qsd_test',
                path='~/bin', compiler='$CC', compile_options='-g -O0',
                delay=True, keep_cc=False)
    assert traj.compile_cmd == '$CC -g -O0 -I~/local/header -o qsd_test '\
                               'qsd_test.cc -L~/local/lib -lqsd'
    assert traj._path == '~/bin'


@mock.patch('qnet.misc.qsd_codegen.compilation_worker',
            return_value='/home/qnet/bin/qsd_test')
def test_compilation_worker(mock_compilation_worker, Sec6_codegen, traj1,
        traj2_10):
    codegen = Sec6_codegen
    codegen.compile(qsd_lib='~/local/lib/libqsd.a',
                qsd_headers='~/local/header', executable='qsd_test',
                path='~/bin', compiler='$CC', compile_options='-g -O0',
                delay=True, keep_cc=False)
    comp_kwargs = {'executable': 'qsd_test', 'path':'~/bin',
                   'cc_code':str(codegen), 'keep_cc': False,
                   'cmd': ['$CC', '-g', '-O0', '-I~/local/header',
                           '-o', 'qsd_test', 'qsd_test.cc', '-L~/local/lib',
                           '-lqsd']}
    operators = OrderedDict([('X1', 'X1.out'), ('X2', 'X2.out'),
                             ('A2', 'A2.out')])
    run_kwargs = {'executable': '/home/qnet/bin/qsd_test',
                  'workdir': '.', 'operators': operators, 'keep': False,
                  'seed': TRAJ1_SEED, 'path': '.'}
    qsd_run_worker = 'qnet.misc.qsd_codegen.qsd_run_worker'
    traj1_ID = traj1.ID
    traj2_10_ID = traj2_10.ID

    traj_IDs = [] # all the IDs we generate by calls to run()

    # The first call to run()
    assert codegen.traj_data is None
    with mock.patch(qsd_run_worker, return_value=traj1) as mock_runner:
        traj_first = codegen.run(seed=TRAJ1_SEED)
        traj_IDs.append(traj_first.ID)
    mock_compilation_worker.assert_called_once_with(comp_kwargs)
    mock_runner.assert_called_once_with(run_kwargs)
    assert codegen.traj_data == traj_first

    # assert that same seed raises early exception
    with mock.patch(qsd_run_worker, return_value=traj2_10) as mock_runner:
        with pytest.raises(ValueError) as exc_info:
            traj = codegen.run(seed=TRAJ1_SEED)
        assert "already in record" in str(exc_info.value)

    # The second call to run()
    with mock.patch(qsd_run_worker, return_value=traj2_10) as mock_runner:
        traj_second = codegen.run(seed=TRAJ2_SEED)
        traj_IDs.append(traj_second.ID)
    run_kwargs['seed'] = TRAJ2_SEED
    mock_runner.assert_called_once_with(run_kwargs)
    assert codegen.traj_data == traj_first + traj_second
    for col, arr in codegen.traj_data.table.items():
        delta = arr - ((traj_first.table[col]+9*traj_second.table[col])/10.0)
        assert np.max(np.abs(delta)) < 1.0e-12

    # Repeated calls to run with auto-seeding
    for call in range(5):
        def side_effect(kwargs):
            return fake_traj(traj1, traj1.new_id(), kwargs['seed'])
        with mock.patch(qsd_run_worker, side_effect=side_effect) \
                as mock_runner:
            traj = codegen.run()
            traj_IDs.append(traj.ID)
    assert len(codegen.traj_data.record) == 7

    # check that bug was fixed where codegen.traj_data was a reference to traj1
    # instead of a copy, thereby modifying the traj1 with the second call
    assert traj1.ID == traj1_ID
    assert traj2_10.ID == traj2_10_ID

