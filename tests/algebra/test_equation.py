"""Test for the Eq class"""
import sympy
from sympy.core.sympify import SympifyError

import pytest

from qnet import (
    OperatorSymbol, Create, Destroy, Eq, latex, IdentityOperator, ZeroOperator)


# These only cover things not already coveraged in the doctest


def test_apply_to_lhs():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = eq0.apply_to_lhs(lambda expr: expr + E0, tag='new')
    assert eq.lhs == H_0 + E0
    assert eq.rhs == eq0.rhs
    assert eq.tag == 'new'


def test_apply_mtd():
    H_0 = OperatorSymbol('H_0', hs=0)
    H = OperatorSymbol('H', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = eq0.apply_mtd('substitute', {H_0: H, E0: 0}, tag='new')
    assert eq.lhs == H
    assert eq.rhs == ω * Create(hs=0) * Destroy(hs=0)
    assert eq.tag == 'new'


def test_eq_copy():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = eq0.copy()
    assert eq == eq0
    assert eq is not eq0


def test_eq_add_const():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = eq0 + E0
    assert eq.lhs == H_0 + E0
    assert eq.rhs == eq0.rhs + E0
    assert eq.tag is None


def test_eq_mult_const():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = 2 * eq0
    assert eq == eq0 * 2
    assert eq.lhs == 2 * eq0.lhs
    assert eq.rhs == 2 * eq0.rhs
    assert eq.tag is None


def test_eq_div_const():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = eq0 / 2
    assert eq.lhs == eq0.lhs / 2
    assert eq.rhs == eq0.rhs / 2
    assert eq.tag is None


def test_eq_equals_const():
    H_0 = OperatorSymbol('H_0', hs=0)
    eq0 = Eq(H_0, IdentityOperator)
    assert eq0 - 1 == ZeroOperator


def test_eq_sub_eq():
    ω, E0 = sympy.symbols('omega, E_0')
    H_0 = OperatorSymbol('H_0', hs=0)
    H_1 = OperatorSymbol('H_1', hs=0)
    mu = OperatorSymbol('mu', hs=0)
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq1 = Eq(H_1, mu + E0, tag='1')
    eq = eq0 - eq1
    assert eq.lhs == H_0 - H_1
    assert eq.rhs == ω * Create(hs=0) * Destroy(hs=0) - mu
    assert eq.tag is None


def test_eq_sub_const():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    eq = eq0 - E0
    assert eq.lhs == H_0 - E0
    assert eq.rhs == ω * Create(hs=0) * Destroy(hs=0)
    assert eq.tag is None


def test_verify_with_func():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    assert eq0.verify(
        lambda expr, mapping: expr.substitute(mapping), {H_0: eq0.rhs})


def test_repr_latex():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    assert eq0._repr_latex_() == latex(eq0)


def test_eq_str():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    assert str(eq0) == "%s = %s    (0)" % (str(eq0.lhs), str(eq0.rhs))


def test_eq_repr():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    assert repr(eq0) == "%s = %s    (0)" % (repr(eq0.lhs), repr(eq0.rhs))


def test_no_sympify():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    with pytest.raises(SympifyError):
        sympy.sympify(eq0)


def test_eq_symbols():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    assert eq0.free_symbols == eq0.all_symbols == set([ω, E0])
    assert eq0.bound_symbols == set()

    eq = eq0.apply(lambda s: s - E0)
    assert eq.all_symbols == eq.free_symbols == eq0.all_symbols

    eq = eq0.apply_to_lhs(lambda s: 1)
    assert eq.all_symbols == eq.free_symbols == eq0.all_symbols

    eq = eq0.apply_to_rhs(lambda s: 1)
    assert eq.all_symbols == eq.free_symbols == set()


def test_eq_substitute():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
    assert eq0.substitute({E0: 0}) == Eq(H_0, ω * Create(hs=0) * Destroy(hs=0))
    assert (
        eq0.substitute({E0: 0}, cont=True) ==
        Eq(H_0, ω * Create(hs=0) * Destroy(hs=0)))


def test_unchanged_apply():
    H_0 = OperatorSymbol('H_0', hs=0)
    ω, E0 = sympy.symbols('omega, E_0')
    eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')

    assert eq0.apply(lambda s: s.expand()) == eq0
    assert eq0.apply(lambda s: s.expand(), cont=True) == eq0
    assert eq0.apply(lambda s: s.expand(), cont=True)._lhs is None

    assert eq0.apply_mtd('expand') == eq0
    assert eq0.apply_mtd('expand', cont=True) == eq0
    assert eq0.apply_mtd('expand', cont=True)._lhs is None
