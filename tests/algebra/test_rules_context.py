import pytest

from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.operator_algebra import (
    OperatorSymbol, ScalarTimesOperator, OperatorPlus, Operator,
    IdentityOperator, OperatorTimes, Commutator)
from qnet.algebra.toolbox.core import temporary_rules
from qnet.algebra.pattern_matching import wc, pattern_head, pattern
from qnet.printing import srepr
from qnet.algebra.core.algebraic_properties import scalars_to_op


def test_no_rules():
    """Test creation of expr when rule application for one or more operation is
    suppressed"""
    A, B = (OperatorSymbol(s, hs=0) for s in ('A', 'B'))
    expr = lambda: Commutator.create(2 * A, 2 * (3 * B))
    myrepr = lambda e: srepr(e, cache={A: 'A', B: 'B'})
    assert (
        myrepr(expr()) ==
        'ScalarTimesOperator(ScalarValue(12), Commutator(A, B))')
    with temporary_rules(ScalarTimesOperator, clear=True):
        assert (
            myrepr(expr()) ==
            'ScalarTimesOperator(ScalarValue(4), '
            'ScalarTimesOperator(ScalarValue(3), Commutator(A, B)))')
    with temporary_rules(Commutator, clear=True):
        assert (
            myrepr(expr()) ==
            'Commutator(ScalarTimesOperator(ScalarValue(2), A), '
            'ScalarTimesOperator(ScalarValue(6), B))')
    with temporary_rules(Commutator, ScalarTimesOperator, clear=True):
        assert (
            myrepr(expr()) ==
            'Commutator(ScalarTimesOperator(ScalarValue(2), A), '
            'ScalarTimesOperator(ScalarValue(2), '
            'ScalarTimesOperator(ScalarValue(3), B)))')
    assert (
        myrepr(expr()) ==
        'ScalarTimesOperator(ScalarValue(12), Commutator(A, B))')


def test_extra_rules():
    """Test creation of expr with extra rules"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    hs_repr = "LocalSpace('h1')"
    rule = (pattern_head(6, a), lambda: b)
    with temporary_rules(ScalarTimesOperator):
        ScalarTimesOperator.add_rule('extra', rule[0], rule[1])
        assert ('extra', rule) in ScalarTimesOperator._rules.items()
        expr = 2*a*3 + 3 * (2*a*3)
        assert expr == 4 * b
    assert rule not in ScalarTimesOperator._rules.values()
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
                pattern(
                    ScalarTimesOperator, -1,
                    pattern(OperatorTimes, B_, A_)),
                ),
            lambda A, B: c)
    with temporary_rules(OperatorPlus):
        OperatorPlus.add_rule('extra', rule[0], rule[1])
        assert ('extra', rule) in OperatorPlus._binary_rules.items()
        expr = 2 * (a * b - b * a + IdentityOperator)
        assert expr == 2 * (c + IdentityOperator)
    assert rule not in OperatorPlus._binary_rules.values()


def test_temporary_simplifications():
    """Test that we can locally modify the simplifications class attrib"""
    assert scalars_to_op in OperatorPlus.simplifications
    with temporary_rules(OperatorPlus):
        OperatorPlus.simplifications.remove(scalars_to_op)
        assert scalars_to_op not in OperatorPlus.simplifications
    assert scalars_to_op in OperatorPlus.simplifications


def test_exception_teardown():
    """Test that teardown works when breaking out due to an exception"""
    class TemporaryRulesException(Exception):
        pass
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    hs_repr = "LocalSpace('h1')"
    rule_name = 'extra'
    rule = (pattern_head(6, a), lambda: b)
    simplifications = OperatorPlus.simplifications
    try:
        with temporary_rules(ScalarTimesOperator, OperatorPlus):
            ScalarTimesOperator.add_rule(rule_name, rule[0], rule[1])
            OperatorPlus.simplifications.remove(scalars_to_op)
            raise TemporaryRulesException
    except TemporaryRulesException:
        assert rule not in ScalarTimesOperator._rules.values()
        assert scalars_to_op in OperatorPlus.simplifications
    finally:
        # Even if this failed we don't want to make a mess for other tests
        try:
            ScalarTimesOperator.del_rules(rule_name)
        except KeyError:
            pass
        OperatorPlus.simplifications = simplifications
