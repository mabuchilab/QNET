import pytest

from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.operator_algebra import (
    OperatorSymbol, ScalarTimesOperator, OperatorPlus, Operator,
    IdentityOperator, OperatorTimes)
from qnet.algebra.toolbox.core import no_rules, extra_binary_rules
from qnet.algebra.toolbox.core import extra_rules
from qnet.algebra.pattern_matching import wc, pattern_head, pattern
from qnet.printing import srepr


def test_no_rules():
    """Test creation of expr when rule application for one or more operation is
    suppressed"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    hs_repr = "LocalSpace('h1')"
    assert (
        srepr(2*a*3 + 3 * (2*a*3)) ==
        "ScalarTimesOperator(ScalarValue(24), OperatorSymbol('a', hs=" +
        hs_repr + "))")
    with no_rules(ScalarTimesOperator):
        expr = 2*a*3 + 3 * (2*a*3)
        print(srepr(expr))
        assert (
            srepr(expr) ==
            "ScalarTimesOperator(ScalarValue(4), "
            "ScalarTimesOperator(ScalarValue(3), "
            "ScalarTimesOperator(ScalarValue(2), OperatorSymbol('a', hs=" +
            hs_repr + "))))")
    with no_rules(OperatorPlus):
        expr = 2*a*3 + 3 * (2*a*3)
        print(srepr(expr))
        assert (
            srepr(expr) ==
            "OperatorPlus(ScalarTimesOperator(ScalarValue(6), "
            "OperatorSymbol('a', hs=" +
            hs_repr + ")), "
            "ScalarTimesOperator(ScalarValue(18), OperatorSymbol('a', hs=" +
            hs_repr + ")))")
    with no_rules(OperatorPlus), no_rules(ScalarTimesOperator):
        expr = 2*a*3 + 3 * (2*a*3)
        print(srepr(expr))
        summand_repr = (
            "ScalarTimesOperator(ScalarValue(3), ScalarTimesOperator("
            "ScalarValue(2), OperatorSymbol('a', hs="+hs_repr+")))")
        assert (srepr(expr) == (
            "OperatorPlus(" + summand_repr +
            ", ScalarTimesOperator(ScalarValue(3), " + summand_repr + "))"))
    assert (srepr(2*a*3 + 3 * (2*a*3)) ==
            "ScalarTimesOperator(ScalarValue(24), "
            "OperatorSymbol('a', hs="+hs_repr+"))")


def test_extra_rules():
    """Test creation of expr with extra rules"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    hs_repr = "LocalSpace('h1')"
    rule = (pattern_head(6, a), lambda: b)
    with extra_rules(ScalarTimesOperator, {'extra': rule}):
        assert ('extra', rule) in ScalarTimesOperator._rules.items()
        expr = 2*a*3 + 3 * (2*a*3)
        assert expr == 4 * b
    assert rule not in ScalarTimesOperator._rules.values()
    with pytest.raises(AttributeError):
        with extra_binary_rules(ScalarTimesOperator, {'extra': rule}):
            expr = 2*a*3 + 3 * (2*a*3)
    assert (srepr(2*a*3 + 3 * (2*a*3)) ==
            "ScalarTimesOperator(ScalarValue(24), "
            "OperatorSymbol('a', hs="+hs_repr+"))")


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
    with extra_binary_rules(OperatorPlus, {'extra': rule}):
        assert ('extra', rule) in OperatorPlus._binary_rules.items()
        expr = 2 * (a * b - b * a + IdentityOperator)
        assert expr == 2 * (c + IdentityOperator)
    assert rule not in OperatorPlus._binary_rules.values()
    with pytest.raises(AttributeError):
        with extra_rules(OperatorPlus, {'extra': rule}):
            expr = 2 * (a * b - b * a + IdentityOperator)
