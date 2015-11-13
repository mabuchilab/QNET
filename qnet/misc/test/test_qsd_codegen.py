from qnet.misc.qsd_codegen import local_ops, QSDCodeGen
from qnet.algebra.circuit_algebra import (
    IdentityOperator, Create, Destroy, LocalOperator, Operator,
    Operation, Circuit, SLH, set_union, TrivialSpace, symbols, sqrt,
    LocalSigma, identity_matrix
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
    Operator A0c = A0.hc();
    TransitionOperator S1_0_1(1,0,1);
    TransitionOperator S1_1_0(1,1,0);""").strip()
    circuit = SLH(identity_matrix(0), [], ad)
    codegen = QSDCodeGen(circuit)
    ob = codegen._operator_basis_lines()
    assert dedent(ob).strip() == dedent("""
    AnnihilationOperator A0(0);
    IdentityOperator Id0(0);
    Operator A0c = A0.hc();
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
    assert scode.strip() == (r'Operator H = ((c) * (A0c) + (c) * (A0) + (chi) '
                             r'* (((A0c * A0c) + (A0 * A0))));')
