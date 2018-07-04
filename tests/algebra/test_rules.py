"""Test *all* of the algebraic rules (the rules in the _rules and _binary_rules
class attributes of all Operation subclasses"""

import logging
from collections import defaultdict

import pytest

import sympy
from sympy import symbols, IndexedBase

import qnet
from qnet import (
    ScalarTimesOperator, OperatorSymbol, LocalSpace,
    no_instance_caching, ZeroOperator, OperatorPlus, KetIndexedSum,
    IdxSym, BasisKet, KetSymbol, StrLabel, FockIndex, IndexOverRange,
    IndexOverFockSpace, ZeroKet, KroneckerDelta)

One = sympy.S.One
Half = One/2
gamma = symbols('gamma')
hs0 = LocalSpace(0)
OpA = OperatorSymbol('A', hs=hs0)
OpB = OperatorSymbol('B', hs=hs0)
OpC = OperatorSymbol('C', hs=hs0)
i = IdxSym('i')
j = IdxSym('j')
a = IndexedBase('a')
ket_sym_hs01 = KetSymbol(
    'Psi', hs=(LocalSpace(0, dimension=2) * LocalSpace(1, dimension=2)))


# The following list defines all the automatic rules we want to test
# (see `test_rule`). Every tuple in the list yields an independent test. Each
# tuple has five elements:
#
# *   The class (``cls``) for which we want to test a rule
# *   The name of the rule we want to test, as a string. This name must be a
#     key in the class' `_rules` or `_binary_rules` class attribute
# *   The positional arguments for the instantiation of the class
#     (as a tuple ``args``)
# *   The keyword arguments for the instantiation of the class
#     (as a dict ``kwargs``)
# *   The expression that is the expected result from
#     ``cls.create(*args, # **kwargs)``
#
# TESTS must contain at least one test for any rule of any class.
TESTS = [
    # Operator Algebra
    (ScalarTimesOperator, 'R001',   # class, rule
        (1, OpA), {},               # args, kwargs
        OpA),                       # expected
    (ScalarTimesOperator, 'R002',
        (0, OpA), {},
        ZeroOperator),
    (ScalarTimesOperator, 'R002',
        (sympy.S.Zero, OpA), {},
        ZeroOperator),
    (ScalarTimesOperator, 'R002',
        (0.0j, OpA), {},
        ZeroOperator),
    (ScalarTimesOperator, 'R003',
        (sympy.symbols('g'), ZeroOperator), {},
        ZeroOperator),
    (ScalarTimesOperator, 'R005',
        (-1, OpA + OpB + OpC), {},
        OperatorPlus(-OpA, -OpB, -OpC)),
    (ScalarTimesOperator, 'R004',
        (2 * gamma, Half * OpA), {},
        ScalarTimesOperator(gamma, OpA)),
    # Circuit Algebra
    # ...
    # State Algebra
    # ...
    (KetIndexedSum, 'R001',
        (KetSymbol(StrLabel(i), hs=0) - KetSymbol(StrLabel(i), hs=0),
            IndexOverFockSpace(i, hs=LocalSpace(0))), {},
        ZeroKet),
    (KetIndexedSum, 'R002',
        (symbols('a') * BasisKet(FockIndex(i), hs=0),
            IndexOverRange(i, 0, 1)), {},
        symbols('a') * KetIndexedSum(
            BasisKet(FockIndex(i), hs=0), IndexOverRange(i, 0, 1))),
]


@pytest.mark.parametrize("cls, rule, args, kwargs, expected", TESTS)
def test_rule(cls, rule, args, kwargs, expected, caplog):
    """Check that for the given `cls` and `rule` name (which must be a key in
    ``cls._rules`` or ``cls._binary_rules``), if we instantiate
    ``cls(*args, **kwargs)``, `rule` is applied and we obtain the `expected`
    result.

    In order to review the log of how all test expressions are created, call
    ``py.test`` as::

        py.test -s --log-cli-level DEBUG ./tests/algebra/test_rules.py
    """
    qnet.algebra.core.abstract_algebra.LOG = True
    qnet.algebra.core.algebraic_properties.LOG = True
    log_marker = "Rule %s.%s" % (cls.__name__, rule)
    print("\n", log_marker)
    with caplog.at_level(logging.DEBUG):
        with no_instance_caching():
            expr = cls.create(*args, **kwargs)
    assert expr == expected
    assert log_marker in caplog.text


def test_all_rules_are_tested():
    """Test that there is a test in TESTS for all rules of any class

    Note that classes that are not mentioned in TESTS at all are not checked.
    """
    tested_rules = defaultdict(list)
    for test in TESTS:
        cls = test[0]
        rule = test[1]
        tested_rules[cls].append(rule)
    for cls in tested_rules.keys():
        if hasattr(cls, '_binary_rules'):
            rules = set(cls._binary_rules.keys())
        elif hasattr(cls, '_rules'):
            rules = set(cls._rules.keys())
        assert set(tested_rules[cls]) == rules
