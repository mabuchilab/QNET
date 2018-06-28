from functools import partial
from textwrap import dedent

import pytest

from sympy import symbols, sqrt, exp, I

from qnet import (
    CircuitSymbol, CIdentity, CircuitZero, CPermutation, SeriesProduct,
    Feedback, SeriesInverse, OperatorSymbol, IdentityOperator, ZeroOperator,
    Create, Destroy, Jz, Jplus, Jminus, Phase, Displace, Squeeze, LocalSigma,
    tr, Adjoint, PseudoInverse, NullSpaceProjector, LocalSpace, TrivialSpace,
    FullSpace, Matrix, KetSymbol, ZeroKet, TrivialKet, BasisKet,
    CoherentStateKet, UnequalSpaces, ScalarTimesKet, OperatorTimesKet, Bra,
    OverlappingSpaces, SpaceTooLargeError, BraKet, KetBra, SuperOperatorSymbol,
    IdentitySuperOperator, ZeroSuperOperator, SuperAdjoint, SPre, SPost,
    SuperOperatorTimesOperator, tree as tree_str,  QuantumDerivative, Scalar,
    ScalarExpression)


def test_circuit_tree():
    """Test tree representation of a circuit algebra expression"""
    A = CircuitSymbol("A_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    expr = Feedback((A << (beta + gamma)) + CIdentity, out_port=2, in_port=0)
    tree = tree_str(expr)
    assert tree == dedent(r'''
    . Feedback(..., out_port=2, in_port=0)
      └─ Concatenation(..., CIdentity)
         ├─ SeriesProduct(A_test, β ⊞ γ)
         │  ├─ CircuitSymbol(A_test, cdim=2)
         │  └─ Concatenation(β, γ)
         │     ├─ CircuitSymbol(β, cdim=1)
         │     └─ CircuitSymbol(γ, cdim=1)
         └─ CIdentity
    ''').strip()
    tree = tree_str(expr, srepr_leaves=True)
    assert tree == dedent(r'''
    . Feedback(..., out_port=2, in_port=0)
      └─ Concatenation(..., CIdentity)
         ├─ SeriesProduct(A_test, β ⊞ γ)
         │  ├─ CircuitSymbol('A_test', cdim=2)
         │  └─ Concatenation(β, γ)
         │     ├─ CircuitSymbol('beta', cdim=1)
         │     └─ CircuitSymbol('gamma', cdim=1)
         └─ CIdentity
    ''').strip()


def test_hilbert_tree():
    """Test tree representation of a Hilbert space algebra expression"""
    H1 = LocalSpace(1)
    H2 = LocalSpace(2)
    tree = tree_str(H1 * H2)
    assert tree == dedent(r'''
    . ProductSpace(ℌ₁, ℌ₂)
      ├─ LocalSpace(1)
      └─ LocalSpace(2)
    ''').strip()


def test_operator_tree():
    """Test tree representation of a operator space algebra expression"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = OperatorSymbol("A", hs=hs1)
    B = OperatorSymbol("B", hs=hs1)
    C = OperatorSymbol("C", hs=hs2)
    gamma = symbols('gamma', positive=True)
    expr = 2 * A - sqrt(gamma) * (B + C)
    tree = tree_str(expr, unicode=False)
    assert tree == dedent(r'''
    . OperatorPlus(2 * A^(q_1), ...)
      +- ScalarTimesOperator(2, A^(q_1))
      |  +- ScalarValue(2)
      |  +- OperatorSymbol(A, hs=H_q_1)
      +- ScalarTimesOperator(-sqrt(gamma), ...)
         +- ScalarValue(-sqrt(gamma))
         +- OperatorPlus(B^(q_1), C^(q_2))
            +- OperatorSymbol(B, hs=H_q_1)
            +- OperatorSymbol(C, hs=H_q_2)
    ''').strip()


def test_ket_tree():
    """Test tree representation of a state algebra expression"""
    hs1 = LocalSpace('q_1', basis=('g', 'e'))
    hs2 = LocalSpace('q_2', basis=('g', 'e'))
    ket_g1 = BasisKet('g', hs=hs1)
    ket_e1 = BasisKet('e', hs=hs1)
    ket_g2 = BasisKet('g', hs=hs2)
    ket_e2 = BasisKet('e', hs=hs2)
    gamma = symbols('gamma', positive=True)
    phase = exp(-I * gamma)
    bell1 = (ket_e1 * ket_g2 - phase * ket_g1 * ket_e2) / sqrt(2)
    bell2 = (ket_e1 * ket_e2 - ket_g1 * ket_g2) / sqrt(2)
    tree = tree_str(KetBra.create(bell1, bell2))
    assert tree == dedent(r'''
    . ScalarTimesOperator(1/2, ...)
      ├─ ScalarValue(1/2)
      └─ KetBra(..., ...)
         ├─ KetPlus(|eg⟩^(q₁⊗q₂), ...)
         │  ├─ TensorKet(|e⟩^(q₁), |g⟩^(q₂))
         │  │  ├─ BasisKet(e, hs=ℌ_q₁)
         │  │  └─ BasisKet(g, hs=ℌ_q₂)
         │  └─ ScalarTimesKet(-exp(-ⅈ γ), |ge⟩^(q₁⊗q₂))
         │     ├─ ScalarValue(-exp(-ⅈ γ))
         │     └─ TensorKet(|g⟩^(q₁), |e⟩^(q₂))
         │        ├─ BasisKet(g, hs=ℌ_q₁)
         │        └─ BasisKet(e, hs=ℌ_q₂)
         └─ KetPlus(|ee⟩^(q₁⊗q₂), -|gg⟩^(q₁⊗q₂))
            ├─ TensorKet(|e⟩^(q₁), |e⟩^(q₂))
            │  ├─ BasisKet(e, hs=ℌ_q₁)
            │  └─ BasisKet(e, hs=ℌ_q₂)
            └─ ScalarTimesKet(-1, |gg⟩^(q₁⊗q₂))
               ├─ ScalarValue(-1)
               └─ TensorKet(|g⟩^(q₁), |g⟩^(q₂))
                  ├─ BasisKet(g, hs=ℌ_q₁)
                  └─ BasisKet(g, hs=ℌ_q₂)
    ''').strip()


def test_sop_operations():
    """Test tree representation of a superoperator algebra expression"""
    hs1 = LocalSpace('q_1', dimension=2)
    hs2 = LocalSpace('q_2', dimension=2)
    A = SuperOperatorSymbol("A", hs=hs1)
    B = SuperOperatorSymbol("B", hs=hs1)
    C = SuperOperatorSymbol("C", hs=hs2)
    gamma = symbols('gamma', positive=True)
    tree = tree_str(2 * A - sqrt(gamma) * (B + C))
    assert (
        tree == dedent(r'''
        . SuperOperatorPlus(2 A^(q₁), ...)
          ├─ ScalarTimesSuperOperator(2, A^(q₁))
          │  ├─ ScalarValue(2)
          │  └─ SuperOperatorSymbol(A, hs=ℌ_q₁)
          └─ ScalarTimesSuperOperator(-√γ, B^(q₁) + C^(q₂))
             ├─ ScalarValue(-√γ)
             └─ SuperOperatorPlus(B^(q₁), C^(q₂))
                ├─ SuperOperatorSymbol(B, hs=ℌ_q₁)
                └─ SuperOperatorSymbol(C, hs=ℌ_q₂)
        ''').strip())


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


def test_derivative_tree(MyScalarFunc):
    s, t, t0 = symbols('s, t, t_0', real=True)
    expr = (  # nested derivative
        MyScalarFunc("f", s, t)
        .diff(s, n=2)
        .diff(t)
        .evaluate_at({t: t0})
        .diff(t0))
    tree = tree_str(expr)
    assert (
        tree == dedent(r'''
        . MyScalarDerivative(..., derivs=((t₀, 1)))
          └─ MyScalarDerivative(..., derivs=..., vals=((t, t₀)))
             └─ ScalarFunc(f, s, t)
        ''').strip())
