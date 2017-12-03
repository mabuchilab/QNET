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

from sympy import symbols, sqrt

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import (
    OperatorSymbol, Commutator, ZeroOperator, Create, Destroy, LocalSigma,
    LocalProjector, IdentityOperator, Jplus, Jz)
from qnet.algebra.toolbox import (
    expand_commutators_leibniz, evaluate_commutators)


def test_disjunct_hs():
    """Test that commutator of objects in disjunt Hilbert spaces is zero"""
    hs1 = LocalSpace("1")
    hs2 = LocalSpace("2")
    alpha, beta = symbols('alpha, beta')
    A = OperatorSymbol('A', hs=hs1)
    B = OperatorSymbol('B', hs=hs2)
    assert Commutator.create(A, B) == ZeroOperator
    assert Commutator.create(alpha, beta) == ZeroOperator
    assert Commutator.create(alpha, B) == ZeroOperator
    assert Commutator.create(A, beta) == ZeroOperator


def test_commutator_hs():
    """Test that commutator is in the correct Hilbert space"""
    hs1 = LocalSpace("1")
    hs2 = LocalSpace("2")
    A = OperatorSymbol('A', hs=hs1)
    B = OperatorSymbol('B', hs=hs2)
    C = OperatorSymbol('C', hs=hs2)
    assert Commutator.create(B, C).space == hs2
    assert Commutator.create(B, A+C).space == hs1 * hs2


def test_pull_out_scalars():
    """Test that scalars are properly pulled out of commutators"""
    hs = LocalSpace("sys")
    A = OperatorSymbol('A', hs=hs)
    B = OperatorSymbol('B', hs=hs)
    alpha, beta = symbols('alpha, beta')
    assert Commutator.create(alpha*A, B) == alpha * Commutator(A, B)
    assert Commutator.create(A, beta*B) == beta * Commutator(A, B)
    assert (Commutator.create(alpha*A, beta*B) ==
            alpha * beta * Commutator(A, B))


def test_commutator_expansion():
    """Test expansion of sums in commutator"""
    hs = LocalSpace("0")
    A = OperatorSymbol('A', hs=hs)
    B = OperatorSymbol('B', hs=hs)
    C = OperatorSymbol('C', hs=hs)
    D = OperatorSymbol('D', hs=hs)
    alpha = symbols('alpha')
    assert Commutator(A+B, C).expand() == Commutator(A, C) + Commutator(B, C)
    assert Commutator(A, B+C).expand() == Commutator(A, B) + Commutator(A, C)
    assert Commutator(A+B, C+D).expand() == (
        Commutator(A, C) + Commutator(A, D) + Commutator(B, C) +
        Commutator(B, D))
    assert Commutator(A+B, C+D+alpha).expand() == (
        Commutator(A, C) + Commutator(A, D) + Commutator(B, C) +
        Commutator(B, D))


def test_diff():
    """Test differentiation of commutators"""
    hs = LocalSpace("0")
    A = OperatorSymbol('A', hs=hs)
    B = OperatorSymbol('B', hs=hs)
    alpha, t = symbols('alpha, t')
    assert Commutator(alpha * t**2 * A, t * B).diff(t) == (
        3 * alpha * t**2 * Commutator(A, B))
    assert Commutator.create(alpha * t**2 * A, t * B).diff(t) == (
        3 * alpha * t**2 * Commutator(A, B))
    assert Commutator(A, B).diff(t) == ZeroOperator


def test_series_expand():
    """Test series expension of commutator"""
    hs = LocalSpace("0")
    A = OperatorSymbol('A', hs=hs)
    B = OperatorSymbol('B', hs=hs)
    a3, a2, a1, a0, b3, b2, b1, b0, t, t0 = symbols(
        'a_3, a_2, a_1, a_0, b_3, b_2, b_1, b_0, t, t_0')
    A_form = (a3 * t**3 + a2 * t**2 + a1 * t + a0) * A
    B_form = (b3 * t**3 + b2 * t**2 + b1 * t + b0) * B
    comm = Commutator.create(A_form, B_form)
    terms = comm.series_expand(t, 0, 2)
    assert terms == (
        a0 * b0 * Commutator(A, B),
        (a0 * b1 + a1 * b0) * Commutator(A, B),
        (a0 * b2 + a1 * b1 + a2 * b0) * Commutator(A, B))

    A_form = (a1 * t + a0) * A
    B_form = (b1 * t + b0) * B
    comm = Commutator.create(A_form, B_form)
    terms = comm.series_expand(t, t0, 1)
    assert terms == (
        ((a0 * b0 + a0 * b1 * t0 + a1 * b0 * t0 + a1 * b1 * t0**2) *
         Commutator(A, B)),
        (a0 * b1 + a1 * b0 + 2 * a1 * b1 * t0) * Commutator(A, B))

    comm = Commutator.create(A, B)
    terms = comm.series_expand(t, t0, 1)
    assert terms == (Commutator(A, B), ZeroOperator)


def test_commutator_oder():
    """Test anti-commutativity of commutators"""
    hs = LocalSpace("0")
    A = OperatorSymbol('A', hs=hs)
    B = OperatorSymbol('B', hs=hs)
    assert Commutator.create(B, A) == -Commutator(A, B)
    a = Destroy(hs=hs)
    a_dag = Create(hs=hs)
    assert Commutator.create(a, a_dag) == -Commutator.create(a_dag, a)


def test_known_commutators():
    """Test that well-known commutators are recognized"""
    hs = LocalSpace("0")
    a = Destroy(hs=hs)
    a_dag = Create(hs=hs)
    assert Commutator.create(a, a_dag) == IdentityOperator
    assert Commutator.create(a_dag, a) == -IdentityOperator

    assert (
        Commutator.create(LocalSigma(1, 0, hs=hs), LocalSigma(0, 1, hs=hs)) ==
        LocalProjector(1, hs=hs) - LocalProjector(0, hs=hs))
    assert (
        Commutator.create(LocalSigma(1, 0, hs=hs), LocalProjector(1, hs=hs)) ==
        (-1 * LocalSigma(1, 0, hs=hs)))
    assert (
        Commutator.create(LocalSigma(1, 0, hs=hs), LocalProjector(0, hs=hs)) ==
        LocalSigma(1, 0, hs=hs))
    assert (
        Commutator.create(LocalSigma(1, 0, hs=hs), Create(hs=hs)) ==
        (-sqrt(2) * LocalSigma(2, 0, hs=hs)))
    assert Commutator.create(Jplus(hs=hs), Jz(hs=hs)) == -Jplus(hs=hs)


def test_commutator_expand_evaluate():
    """Test expansion and evaluation of commutators"""
    hs = LocalSpace("0")
    A = OperatorSymbol('A', hs=hs)
    B = OperatorSymbol('B', hs=hs)
    C = OperatorSymbol('C', hs=hs)
    D = OperatorSymbol('D', hs=hs)
    E = OperatorSymbol('E', hs=hs)
    expr = Commutator(A, B*C*D*E)
    res = (B * C * D * Commutator(A, E) + B * C * Commutator(A, D) * E +
           B * Commutator(A, C) * D * E + Commutator(A, B) * C * D * E)
    assert expand_commutators_leibniz(expr) == res
    assert evaluate_commutators(expr) == (
        A * B * C * D * E - B * C * D * E * A)
    assert evaluate_commutators(res).expand() == (
        A * B * C * D * E - B * C * D * E * A)

    assert expand_commutators_leibniz(expr, expand_expr=False) == (
        B * (C * (D * Commutator(A, E) + Commutator(A, D) * E) +
             Commutator(A, C) * D * E) + Commutator(A, B) * C * D * E)

    expr = Commutator(A*B*C, D)
    assert expand_commutators_leibniz(expr) == (
        A*B*Commutator(C, D) + A*Commutator(B, D)*C + Commutator(A, D)*B*C)

    expr = Commutator(A*B, C*D)
    assert expand_commutators_leibniz(expr) == (
        A * Commutator(B, C) * D + C * A * Commutator(B, D) +
        C * Commutator(A, D) * B + Commutator(A, C) * B * D)
