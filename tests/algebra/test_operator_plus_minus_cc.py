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
# Copyright (C) 2016 Michael Goerz
#
###########################################################################

import pytest

from sympy import Symbol, I

from qnet.algebra.abstract_algebra import extra_binary_rules
from qnet.algebra.operator_algebra import (
        Create, Destroy, LocalSigma, create_operator_pm_cc, OperatorPlus,
        OperatorPlusMinusCC, expand_operator_pm_cc)
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.printing import srepr


def test_simple_cc():
    """Test that we can find complex conjugates in a sum directly"""
    hs_c = LocalSpace('c', dimension=3)
    hs_q = LocalSpace('q1', basis=('g', 'e'))
    Delta_1 = Symbol('Delta_1')
    Omega_1 = Symbol('Omega_1')
    g_1 = Symbol('g_1')
    a = Destroy(hs=hs_c)
    a_dag = Create(hs=hs_c)
    sig_p = LocalSigma('e', 'g', hs=hs_q)
    sig_m = LocalSigma('g', 'e', hs=hs_q)
    coeff = (-I / 2) * (Omega_1 * g_1 / Delta_1)
    jc_expr= coeff * (a * sig_p  - a_dag * sig_m)

    with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
        simplified = jc_expr.simplify()
    assert simplified == coeff * OperatorPlusMinusCC(a * sig_p, sign=-1)
    assert (srepr(simplified.term) ==
            "OperatorPlusMinusCC(OperatorTimes(Destroy(hs=LocalSpace('c', "
            "dimension=3)), LocalSigma('e', 'g', hs=LocalSpace('q1', "
            "basis=('g', 'e')))), sign=-1)")
    expanded = simplified.simplify(rules=expand_operator_pm_cc())
    assert expanded == jc_expr


def test_scalar_coeff_cc():
    """Test that we can find complex conjugates in a sum of
    ScalarTimesOperator"""
    hs_1 = LocalSpace('q1', basis=('g', 'e'))
    hs_2 = LocalSpace('q2', basis=('g', 'e'))
    kappa = Symbol('kappa', real=True)
    a1 = Destroy(hs=hs_1)
    a2 = Destroy(hs=hs_2)

    jc_expr = I/2 * (2*kappa * (a1.dag() * a2) - 2*kappa * (a1 * a2.dag()))

    with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
        simplified = jc_expr.simplify()
    assert (simplified ==
            I * kappa * OperatorPlusMinusCC(a1.dag() * a2, sign=-1))
    expanded = simplified.simplify(rules=expand_operator_pm_cc())
    assert expanded == I * kappa * (a1.dag() * a2 - a1 * a2.dag())
