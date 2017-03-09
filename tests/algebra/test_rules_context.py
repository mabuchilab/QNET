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
    IdentityOperator, OperatorTimes)
from qnet.algebra.abstract_algebra import (
    no_rules, extra_rules, extra_binary_rules)
from qnet.algebra.pattern_matching import wc, pattern_head, pattern
from qnet.printing import srepr


def test_no_rules():
    """Test creation of expr when rule application for one or more operation is
    suppressed"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    hs_repr = "LocalSpace('h1')"
    assert (srepr(2*a*3 + 3 * (2*a*3)) ==
            "ScalarTimesOperator(24, ""OperatorSymbol('a', hs="+hs_repr+"))")
    with no_rules(ScalarTimesOperator):
        expr = 2*a*3 + 3 * (2*a*3)
        print(srepr(expr))
        assert (srepr(expr) ==
            "ScalarTimesOperator(4, ScalarTimesOperator(3, "
            "ScalarTimesOperator(2, OperatorSymbol('a', hs="+hs_repr+"))))")
    with no_rules(OperatorPlus):
        expr = 2*a*3 + 3 * (2*a*3)
        print(srepr(expr))
        assert (srepr(expr) ==
            "OperatorPlus(ScalarTimesOperator(6, OperatorSymbol('a', hs=" +
            hs_repr + ")), ScalarTimesOperator(18, OperatorSymbol('a', hs=" +
            hs_repr + ")))")
    with no_rules(OperatorPlus), no_rules(ScalarTimesOperator):
        expr = 2*a*3 + 3 * (2*a*3)
        print(srepr(expr))
        summand_repr = ("ScalarTimesOperator(3, ScalarTimesOperator("
                        "2, OperatorSymbol('a', hs="+hs_repr+")))")
        assert (srepr(expr) == ("OperatorPlus(" + summand_repr +
                                ", ScalarTimesOperator(3, " +
                                summand_repr + "))"))
    assert (srepr(2*a*3 + 3 * (2*a*3)) ==
            "ScalarTimesOperator(24, ""OperatorSymbol('a', hs="+hs_repr+"))")


def test_extra_rules():
    """Test creation of expr with extra rules"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    hs_repr = "LocalSpace('h1')"
    rule = (pattern_head(6, a), lambda: b)
    with extra_rules(ScalarTimesOperator, [rule, ]):
        assert rule in ScalarTimesOperator._rules
        expr = 2*a*3 + 3 * (2*a*3)
        assert expr == 4 * b
    assert rule not in ScalarTimesOperator._rules
    with pytest.raises(AttributeError):
        with extra_binary_rules(ScalarTimesOperator, [rule, ]):
            expr = 2*a*3 + 3 * (2*a*3)
    assert (srepr(2*a*3 + 3 * (2*a*3)) ==
            "ScalarTimesOperator(24, ""OperatorSymbol('a', hs="+hs_repr+"))")


def test_extra_binary_rules():
    """Test creation of expr with extra binary rules"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    c = OperatorSymbol("c", hs=h1)
    A_ = wc('A', head=Operator)
    B_ = wc('B', head=Operator)
    rule = (pattern_head(
                pattern(OperatorTimes, A_, B_),
                pattern(ScalarTimesOperator, -1,
                    pattern(OperatorTimes, B_, A_)),
                ),
            lambda A, B: c)
    with extra_binary_rules(OperatorPlus, [rule, ]):
        assert rule in OperatorPlus._binary_rules
        expr = 2 * (a * b - b * a + IdentityOperator)
        assert expr == 2 * (c + IdentityOperator)
    assert rule not in OperatorPlus._binary_rules
    with pytest.raises(AttributeError):
        with extra_rules(OperatorPlus, [rule, ]):
            expr = 2 * (a * b - b * a + IdentityOperator)
