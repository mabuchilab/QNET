from sympy import symbols, Indexed
import unittest
from collections import OrderedDict


from qnet import (
    Operation, ScalarTimesQuantumExpression, IndexedSum, CannotSimplify,
    IndexOverRange, IdxSym, pattern, pattern_head, wc, Operator,
    LocalProjector, LocalOperator, OperatorTimes, OperatorSymbol, Commutator,
    ZeroOperator, OperatorPlus, Displace, LocalSpace, StrLabel, Sum,
    OperatorIndexedSum)

from qnet.algebra.core.algebraic_properties import (
    assoc, assoc_indexed, idem,
    orderby, filter_neutral, match_replace, match_replace_binary,
    indexed_sum_over_const)
from qnet.utils.ordering import expr_order_key

import pytest


def test_match_replace_binary_complete():
    """Test that replace_binary works correctly for a non-trivial case"""
    x, y, z, alpha = symbols('x y z alpha')
    hs = LocalSpace('f')
    ops = [LocalProjector(0, hs=hs),
           Displace(-alpha, hs=hs),
           Displace(alpha, hs=hs),
           LocalProjector(0, hs=hs)]
    res = OperatorTimes.create(*ops)
    assert res == LocalProjector(0, hs=hs)


def test_apply():
    """Test the apply method"""

    A = OperatorSymbol('A', hs=0)

    def raise_to_power(x, y):
        return x**y

    def plus_n(expr, *, n):
        return expr + n

    assert (
        A
        .apply(raise_to_power, 2)
        .apply(plus_n, n=1)
        == A**2 + 1)


@pytest.mark.parametrize("cls", [Commutator, OperatorTimes])
def test_rule_manipulation(cls):
    """Test that manipulating algebraic rules works as expected"""
    n_rules = len(cls.rules())
    assert n_rules > 0
    with pytest.raises(AttributeError):
        cls.rules(attr='_rules')          # one of these ...
        cls.rules(attr='_binary_rules')   # ... raises the exception
    with pytest.raises(AttributeError):
        cls.rules(attr='bogus')
    try:
        orig_rules = cls._rules.copy()
    except AttributeError:
        orig_rules = cls._binary_rules.copy()

    cls.del_rules()
    assert len(cls.rules()) == 0
    for (name, (pat, replacement)) in orig_rules.items():
        cls.add_rule(name, pat, replacement)
    assert len(cls.rules()) == n_rules
    with pytest.raises(AttributeError):
        cls.del_rules(attr='bogus')

    for name in orig_rules.keys():
        cls.del_rules(name)
    assert len(cls.rules()) == 0
    for (name, (pat, replacement)) in orig_rules.items():
        cls.add_rule(name, pat, replacement)
    assert len(cls.rules()) == n_rules

    cls.del_rules(*cls.rules())
    assert len(cls.rules()) == 0
    for (name, (pat, replacement)) in orig_rules.items():
        cls.add_rule(name, pat, replacement)
    assert len(cls.rules()) == n_rules


def test_no_rule_manipulation():
    """Test that manipulating the rules of an object that has no rules raises
    the appropriate exceptions"""
    assert hasattr(LocalOperator, 'simplifications')
    assert match_replace not in LocalOperator.simplifications
    assert match_replace_binary not in LocalOperator.simplifications
    assert len(LocalOperator.rules()) == 0
    with pytest.raises(TypeError) as exc_info:
        LocalOperator.add_rule(None, None, None)
    assert "does not have match_replace" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        LocalOperator.del_rules('R000')
    assert "does not have match_replace" in str(exc_info.value)


def test_rule_manipulation_exceptions():
    """Test that manipulating rules incorrectly raises the appropriate
    exceptions"""
    A = wc("A", head=Operator)
    assert 'R001' in Commutator.rules()
    with pytest.raises(KeyError):
        Commutator.del_rules('XXX')
    with pytest.raises(TypeError) as exc_info:
        Commutator.add_rule(None, pattern_head(A, A), lambda A: ZeroOperator)
    assert "'None' is not a string" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        Commutator.add_rule('E001', None, lambda A: 0)
    assert "Pattern in 'E001' is not a Pattern instance" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        Commutator.add_rule('E001', pattern(Operator, A, A), lambda A: 0)
    assert "'E001' does not match a ProtoExpr" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        Commutator.add_rule('E001', pattern_head(A, A), None)
    assert "replacement in 'E001' is not callable" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        Commutator.add_rule('E001', pattern_head(A, A), ZeroOperator)
    assert (
        'arguments () of replacement function differ from the wildcard '
        'names (A) in pattern' in str(exc_info.value))
    with pytest.raises(ValueError) as exc_info:
        Commutator.add_rule('R001', pattern_head(A, A), lambda A: ZeroOperator)
    assert "rule already exists" in str(exc_info.value)
    assert 'R002' in OperatorTimes.rules()
    with pytest.raises(ValueError) as exc_info:
        OperatorTimes.add_rule('R002', None, None)
    assert "rule already exists" in str(exc_info.value)


def test_show_rules(capsys):
    OperatorTimes.show_rules('R002', 'R004')
    out = capsys.readouterr()[0]
    assert 'R002' in out
    assert 'R004' in out
    assert 'R001' not in out
    with pytest.raises(AttributeError):
        OperatorTimes.show_rules('R002', 'R004', attr='bogus')

    LocalOperator.show_rules()  # has no rules
    out = capsys.readouterr()[0]
    assert out == ''


def test_nested_doit():
    """Test a complete doit-invocation on a nested expression"""
    # This tests some tricky edge cases, augmenting the doctest

    def A(i):
        if isinstance(i, IdxSym):
            return OperatorSymbol(StrLabel(Indexed('A', i)), hs=0)
        else:
            return OperatorSymbol("A_%s" % i, hs=0)

    i, j = symbols('i, j', cls=IdxSym)

    expr = Sum(i, 1, 3)(Sum(j, 1, 2)(Commutator(A(i), A(j))))
    assert expr.doit(max_terms=2, recursive=False) == Commutator(A(1), A(2))

    # testing the "tail" of the recursion
    assert (
        expr.doit(max_terms=2, classes=([OperatorIndexedSum])) ==
        Commutator(A(1), A(2)))
    assert expr.doit(max_terms=2) == A(1) * A(2) - A(2) * A(1)

    expr = Sum(i, 1, 3)(Sum(j, 1, 3)(Commutator(A(i), A(j))))
    assert expr.doit() == ZeroOperator
    # testing that `indices=(i, )` does not throw an error in the recursion,
    # when i no longer occurs in the sum
    assert expr.doit(indices=(i, )) == Sum(j, 1, 3)(
        OperatorPlus(
             A(1) * A(j), A(2) * A(j), A(3) * A(j),
             -A(j) * A(1), -A(j) * A(2), -A(j) * A(3)))


def test_create_with_mutable_args():
    hs = LocalSpace.create(0, basis=['g', 'e'], local_identifiers={'Destroy': 'b'})
    assert hs.instance_caching
