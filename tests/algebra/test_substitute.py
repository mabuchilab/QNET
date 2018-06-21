from sympy import symbols
import pytest

from qnet.algebra.core.abstract_algebra import substitute
from qnet.algebra.core.exceptions import BasisNotSetError
from qnet.algebra.core.matrix_algebra import Matrix
from qnet.algebra.core.operator_algebra import (
    IdentityOperator, II, OperatorSymbol)
from qnet.algebra.library.fock_operators import Destroy
from qnet.algebra.core.hilbert_space_algebra import LocalSpace


@pytest.fixture
def H_JC():
    hil_a = LocalSpace('A')
    hil_b = LocalSpace('B')
    a = Destroy(hs=hil_a)
    a_dag = a.dag()
    b = Destroy(hs=hil_b)
    b_dag = b.dag()
    omega_a, omega_b, g = symbols('omega_a, omega_b, g')
    H = (omega_a * a_dag * a + omega_b * b_dag * b +
         2 * g * (a_dag * b + b_dag * a))
    return H


def test_substitute_basis(H_JC):
    """"Test that we can assign an expression's Hilbert space a basis"""
    H = H_JC
    with pytest.raises(BasisNotSetError):
        H.space.dimension

    hs_mapping = {
        LocalSpace('A'): LocalSpace('A', basis=('g', 'e')),
        LocalSpace('B'): LocalSpace('B', dimension=10),
    }

    H2 = H.substitute(hs_mapping)
    assert H2.space.dimension == 20

    H2 = substitute(H, hs_mapping)
    assert H2.space.dimension == 20


def test_substitute_numvals(H_JC):
    """Test that we can substitute in numbers for scalar coefficients"""
    omega_a, omega_b, g = symbols('omega_a, omega_b, g')

    num_vals = {
        omega_a: 0.2,
        omega_b: 0,
        g: 1,
    }

    hil_a = LocalSpace('A')
    hil_b = LocalSpace('B')
    a = Destroy(hs=hil_a)
    a_dag = a.dag()
    b = Destroy(hs=hil_b)
    b_dag = b.dag()
    H2_expected = 0.2 * a_dag * a + 2 * (a_dag * b + b_dag * a)

    H2 = H_JC.substitute(num_vals)
    assert H2 == H2_expected

    H2 = substitute(H_JC, num_vals)
    assert H2 == H2_expected


def test_substitute_str(H_JC):
    """Test that we can substitute e.g. label strings"""
    H2 = H_JC.substitute({'A': '1', 'B': '2'})
    hs_mapping = {
        LocalSpace('A'): LocalSpace('1'),
        LocalSpace('B'): LocalSpace('2'),
    }
    assert H2 == H_JC.substitute(hs_mapping)


def test_substitute_sympy_formula(H_JC):
    """Test that we can replace sympy symbols with other sympy formulas"""
    omega_a, omega_b, g = symbols('omega_a, omega_b, g')
    Delta_a, Delta_b, delta, kappa = symbols('Delta_a, Delta_b, delta, kappa')
    hil_a = LocalSpace('A')
    hil_b = LocalSpace('B')
    a = Destroy(hs=hil_a)
    a_dag = a.dag()
    b = Destroy(hs=hil_b)
    b_dag = b.dag()
    mapping = {
        omega_a: Delta_a,
        omega_b: Delta_b,
        g: kappa / (2 * delta)
    }
    H2_expected = (
        Delta_a * a_dag * a + Delta_b * b_dag * b +
        (kappa / delta) * (a_dag * b + b_dag * a))

    H2 = H_JC.substitute(mapping)
    assert H2 == H2_expected

    H2 = substitute(H_JC, mapping)
    assert H2 == H2_expected


def test_substitute_total_expression(H_JC):
    """Test that we can replace the entire expr with another expression"""
    C = OperatorSymbol('C', hs=H_JC.space)
    assert H_JC.substitute({H_JC: C}) == C
    assert substitute(H_JC, {H_JC: C}) == C


def test_substitute_symbol_not_in_expr(H_JC):
    """Test that we if a symbol in the mapping dict does not occur in the expr,
    we don't get an error, but leaves the expr unchanged"""
    x = symbols('x')
    assert H_JC.substitute({x: 0}) == H_JC
    assert substitute(H_JC, {x: 0}) == H_JC


def test_substitute_sub_expr(H_JC):
    """Test that we can replace non-atomic sub-expressions"""
    hil_a = LocalSpace('A')
    hil_b = LocalSpace('B')
    omega_a, omega_b, g = symbols('omega_a, omega_b, g')
    a = Destroy(hs=hil_a)
    a_dag = a.dag()
    b = Destroy(hs=hil_b)
    b_dag = b.dag()
    n_op_a = OperatorSymbol('n', hs=hil_a)
    n_op_b = OperatorSymbol('n', hs=hil_b)
    x_op = OperatorSymbol('x', hs=H_JC.space)
    mapping = {
        a_dag * a: n_op_a,
        b_dag * b: n_op_b,
        (a_dag * b + b_dag * a): x_op + x_op.dag()
    }
    H2_expected = (omega_a * n_op_a + omega_b * n_op_b +
                   2 * g * (x_op + x_op.dag()))

    H2 = H_JC.substitute(mapping)
    assert H2 == H2_expected

    H2 = substitute(H_JC, mapping)
    assert H2 == H2_expected


def test_substitute_matrix(H_JC):
    """Test that we can substitute in a Matrix (element-wise)"""
    M = Matrix([[H_JC, IdentityOperator], [IdentityOperator, H_JC]])
    IM = Matrix([[IdentityOperator, IdentityOperator],
                 [IdentityOperator, IdentityOperator]])
    assert M.substitute({H_JC: IdentityOperator}) == M.substitute({M: IM})
    assert substitute(M, {H_JC: IdentityOperator}) == substitute(M, {M: IM})


def test_substitute_sympy():
    """Test that the sustitute function can directly modify sympy
    expressions"""
    g, kappa = symbols('g, kappa')
    assert substitute(g**2/2, {g**2: kappa}) == kappa / 2


def test_singleton_substitute():
    """Test that calling the substitute method on a Singleton returns the
    Singleton"""
    assert II.substitute({}) is II
