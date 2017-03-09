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

from textwrap import dedent

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
from qnet.printing.tree import tree_str

import pytest


def test_circuit_tree():
    """Test tree representation of a circuit algebra expression"""
    A = CircuitSymbol("A_test", cdim=2)
    beta = CircuitSymbol("beta", cdim=1)
    gamma = CircuitSymbol("gamma", cdim=1)
    tree = tree_str(Feedback((A << (beta + gamma)) + CIdentity,
                             out_port=2, in_port=0))
    assert tree == dedent(r'''
    . Feedback(..., out_port=2, in_port=0)
      └─ Concatenation(..., cid(1))
         ├─ SeriesProduct(A_test, β ⊞ γ)
         │  ├─ CircuitSymbol(A_test, 2)
         │  └─ Concatenation(β, γ)
         │     ├─ CircuitSymbol(beta, 1)
         │     └─ CircuitSymbol(gamma, 1)
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
      |  +- 2
      |  +- OperatorSymbol(A, hs=H_q_1)
      +- ScalarTimesOperator(-sqrt(gamma), ...)
         +- -sqrt(gamma)
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
      ├─ 1/2
      └─ KetBra(..., ...)
         ├─ KetPlus(|e,g⟩_(q₁⊗q₂), ...)
         │  ├─ TensorKet(|e⟩_(q₁), |g⟩_(q₂))
         │  │  ├─ BasisKet(e, hs=ℌ_q₁)
         │  │  └─ BasisKet(g, hs=ℌ_q₂)
         │  └─ ScalarTimesKet(-exp(-I*γ), |g,e⟩_(q₁⊗q₂))
         │     ├─ -exp(-I*gamma)
         │     └─ TensorKet(|g⟩_(q₁), |e⟩_(q₂))
         │        ├─ BasisKet(g, hs=ℌ_q₁)
         │        └─ BasisKet(e, hs=ℌ_q₂)
         └─ KetPlus(|e,e⟩_(q₁⊗q₂), -|g,g⟩_(q₁⊗q₂))
            ├─ TensorKet(|e⟩_(q₁), |e⟩_(q₂))
            │  ├─ BasisKet(e, hs=ℌ_q₁)
            │  └─ BasisKet(e, hs=ℌ_q₂)
            └─ ScalarTimesKet(-1, |g,g⟩_(q₁⊗q₂))
               ├─ -1
               └─ TensorKet(|g⟩_(q₁), |g⟩_(q₂))
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
        . SuperOperatorPlus(2 * A^(q₁), ...)
          ├─ ScalarTimesSuperOperator(2, A^(q₁))
          │  ├─ 2
          │  └─ SuperOperatorSymbol(A, hs=ℌ_q₁)
          └─ ScalarTimesSuperOperator(-√γ, B^(q₁) + C^(q₂))
             ├─ -sqrt(gamma)
             └─ SuperOperatorPlus(B^(q₁), C^(q₂))
                ├─ SuperOperatorSymbol(B, hs=ℌ_q₁)
                └─ SuperOperatorSymbol(C, hs=ℌ_q₂)
        ''').strip() or
        # The sympy printer doesn't always give exactly the same result,
        # depending on context
        tree == dedent(r'''
        . SuperOperatorPlus(2 * A^(q₁), ...)
          ├─ ScalarTimesSuperOperator(2, A^(q₁))
          │  ├─ 2
          │  └─ SuperOperatorSymbol(A, hs=ℌ_q₁)
          └─ ScalarTimesSuperOperator(-sqrt(γ), B^(q₁) + C^(q₂))
             ├─ -sqrt(gamma)
             └─ SuperOperatorPlus(B^(q₁), C^(q₂))
                ├─ SuperOperatorSymbol(B, hs=ℌ_q₁)
                └─ SuperOperatorSymbol(C, hs=ℌ_q₂)
        ''').strip())
