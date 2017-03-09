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

import pytest

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import (
    OperatorSymbol, ScalarTimesOperator, OperatorPlus, Operator,
    OperatorTimes)
from qnet.algebra.abstract_algebra import (
     extra_binary_rules, simplify, CannotSimplify)
from qnet.algebra.pattern_matching import wc, pattern_head, pattern
from qnet.printing import srepr


def test_simplify():
    """Test simplification of expr according to manual rules"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    c = OperatorSymbol("c", hs=h1)
    d = OperatorSymbol("d", hs=h1)

    expr = 2 * (a * b * c - b * c * a)

    A_ = wc('A', head=Operator)
    B_ = wc('B', head=Operator)
    C_ = wc('C', head=Operator)

    def b_times_c_equal_d(B, C):
        if (B.identifier == 'b' and C.identifier =='c'):
            return d
        else:
            raise CannotSimplify

    with extra_binary_rules(
            OperatorTimes, [(pattern_head(B_, C_), b_times_c_equal_d), ]):
        new_expr = simplify(expr)

    commutator_rule = (
            pattern(OperatorPlus,
                pattern(OperatorTimes, A_, B_),
                pattern(ScalarTimesOperator,
                        -1, pattern(OperatorTimes, B_, A_))),
            lambda A, B: OperatorSymbol(
                "Commut%s%s" % (A.identifier.upper(), B.identifier.upper()),
                hs=A.space)
            )
    assert commutator_rule[0].match(new_expr.term)

    with extra_binary_rules(
            OperatorTimes, [(pattern_head(B_, C_), b_times_c_equal_d), ]):
        new_expr = simplify(expr, [commutator_rule, ])
    assert (srepr(new_expr) ==
            "ScalarTimesOperator(2, OperatorSymbol('CommutAD', "
            "hs=LocalSpace('h1')))")
