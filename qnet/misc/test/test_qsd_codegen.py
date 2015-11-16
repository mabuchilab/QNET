from qnet.misc.qsd_codegen import local_ops, QSDCodeGen, QSDOperator
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, SLH, set_union, TrivialSpace, symbols, sqrt,
    LocalSigma, identity_matrix, I
)
from textwrap import dedent
from qnet.circuit_components.pseudo_nand_cc import PseudoNAND
import pytest

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


def test_qsd_codegen_operator_basis():
    a = Destroy(1)
    ad = a.dag()
    s = LocalSigma(2, 1, 0)
    sd = s.dag()
    circuit = SLH(identity_matrix(0), [], a*ad + s + sd)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    AnnihilationOperator A0(0);
    IdentityOperator Id0(0);
    IdentityOperator Id1(1);
    Operator Ad0 = A0.hc();
    Operator Id = Id0*Id1;
    TransitionOperator S1_0_1(1,0,1);
    TransitionOperator S1_1_0(1,1,0);""").strip()
    circuit = SLH(identity_matrix(0), [], ad)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    AnnihilationOperator A0(0);
    IdentityOperator Id0(0);
    Operator Ad0 = A0.hc();
    Operator Id = Id0;
    """).strip()


def test_qsd_codegen_parameters():
    k = symbols("kappa", positive=True)
    x = symbols("chi", real=True)
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
    # TODO: test missing values


def test_qsd_codegen_hamiltonian():
    k = symbols("kappa", positive=True)
    x = symbols("chi", real=True)
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

    S = QSDOperator('TransitionOperator', 'S1_0_1', '(1,0,1)')
    assert S.instantiation == 'TransitionOperator S1_0_1(1,0,1);'

    Ad0 = QSDOperator('Operator', 'Ad0', '= A0.hc()')
    assert Ad0.instantiator == '= A0.hc()'
    assert Ad0.instantiation == 'Operator Ad0 = A0.hc();'

    # spaces around the '=' should be stripped out
    Ad0_2 = QSDOperator('Operator', 'Ad0', '  =   A0.hc()')
    assert Ad0_2.instantiator == Ad0.instantiator
    Ad0_2 = QSDOperator('Operator', 'Ad0', '=A0.hc()')
    assert Ad0_2.instantiator == Ad0.instantiator

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


def test_qsd_codegen_nlinkblads():
    k = symbols("kappa", positive=True)
    x = symbols("chi", real=True)
    c = symbols("c",real=True)
    a = Destroy(1)
    ad = a.dag()
    s = LocalSigma(2, 1, 0)
    sd = s.dag()

    circuit = SLH(identity_matrix(0), [], a*ad + s + sd)
    codegen = QSDCodeGen(circuit, num_vals={x: 2., c: 1, k: 2})
    result = codegen.generate_code()
    assert "const int nL = 0;" in codegen.generate_code()

    H = x * (a * a + a.dag() * a.dag()) + (c * a.dag() + c.conjugate() * a)
    L = sqrt(k) * a
    circuit = SLH(identity_matrix(1), [L], H)
    codegen = QSDCodeGen(circuit, num_vals={x: 2., c: 1, k: 2})
    assert "const int nL = 1;" in codegen.generate_code()


def test_qsd_codegen_lindblads():
    E      = symbols("E", positive=True)
    chi    = symbols("chi", real=True)
    omega  = symbols("omega",real=True)
    eta    = symbols("eta",real=True)
    gamma1 = symbols("gamma1", positive=True)
    gamma2 = symbols("gamma2", positive=True)
    kappa  = symbols("kamma", positive=True)
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

    slh = SLH(identity_matrix(3), Lindblads, H)

    codegen = QSDCodeGen(circuit=slh,
              num_vals={E: 20.0, chi: 0.4, omega: -0.7, eta: 0.001,
                        gamma1: 1.0, gamma2: 1.0, kappa: 0.1})
    scode = codegen._lindblads_lines()
    assert dedent(scode).strip() == dedent(r'''
    Operator L[nL]={
    (sqrt(2)*sqrt(gamma1)) * (A0),
    (sqrt(2)*sqrt(gamma2)) * (A1),
    (sqrt(2)*sqrt(kamma)) * (S2_0_1)
    }''').strip()

