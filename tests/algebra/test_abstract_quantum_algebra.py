from collections import OrderedDict
from qnet import (
    One, Zero, ZeroOperator, IdentityOperator, ZeroSuperOperator,
    IdentitySuperOperator, ZeroKet, TrivialKet, FullSpace, TrivialSpace,
    CIdentity, CircuitZero, IdxSym, BasisKet, OperatorSymbol, FockIndex,
    KetIndexedSum, OperatorIndexedSum, StrLabel, LocalSpace,
    IndexOverList, IndexOverFockSpace, IndexOverRange, Sum, ScalarExpression,
    QuantumDerivative, OperatorDerivative, Scalar, ScalarTimes, Create,
    Destroy)
import sympy
from sympy import IndexedBase, symbols

import pytest


def test_neutral_elements():
    """test the properties of the neutral elements in the quantum algebras.
    This tests the resolution of #63

    *Only* the scalar neutral elements compare to and hash as 0 and 1. The
    neutral elements of all other algebras are "unique" and don't compare to 0
    and 1. Elements of a quantum algebra have an is_zero attribute
    """
    assert One == 1
    assert One is not 1
    assert hash(One) == hash(1)
    assert Zero == 0
    assert Zero is not 0
    assert hash(Zero) == hash(0)
    assert Zero.is_zero

    assert IdentityOperator != 1
    assert hash(IdentityOperator) != hash(1)
    assert ZeroOperator != 0
    assert hash(ZeroOperator) != hash(0)
    assert ZeroOperator.is_zero

    assert IdentitySuperOperator != 1
    assert hash(IdentitySuperOperator) != hash(1)
    assert ZeroSuperOperator != 0
    assert hash(ZeroSuperOperator) != hash(0)
    assert ZeroSuperOperator.is_zero

    assert TrivialKet != 1
    assert hash(TrivialKet) != hash(1)
    assert ZeroKet != 0
    assert hash(ZeroKet) != hash(0)
    assert ZeroKet.is_zero

    #  the remainder are not quantum algebra elements, to they don't have
    #  is_zero
    assert FullSpace != 1
    assert hash(FullSpace) != hash(1)
    assert TrivialSpace != 0
    assert hash(TrivialSpace) != hash(0)

    assert CIdentity != 1
    assert hash(CIdentity) != hash(1)
    assert CircuitZero != 0
    assert hash(CircuitZero) != hash(0)


def test_sum_instantiator():
    """Test use of Sum instantiator"""
    i = IdxSym('i')
    j = IdxSym('j')
    ket_i = BasisKet(FockIndex(i), hs=0)
    ket_j = BasisKet(FockIndex(j), hs=0)
    A_i = OperatorSymbol(StrLabel(IndexedBase('A')[i]), hs=0)
    hs0 = LocalSpace('0')

    sum = Sum(i)(ket_i)
    ful = KetIndexedSum(ket_i, IndexOverFockSpace(i, hs=hs0))
    assert sum == ful
    assert sum == Sum(i, hs0)(ket_i)
    assert sum == Sum(i, hs=hs0)(ket_i)

    sum = Sum(i, 1, 10)(ket_i)
    ful = KetIndexedSum(ket_i, IndexOverRange(i, 1, 10))
    assert sum == ful
    assert sum == Sum(i, 1, 10, 1)(ket_i)
    assert sum == Sum(i, 1, to=10, step=1)(ket_i)
    assert sum == Sum(i, 1, 10, step=1)(ket_i)

    sum = Sum(i, (1, 2, 3))(ket_i)
    ful = KetIndexedSum(ket_i, IndexOverList(i, (1, 2, 3)))
    assert sum == KetIndexedSum(ket_i, IndexOverList(i, (1, 2, 3)))
    assert sum == Sum(i, [1, 2, 3])(ket_i)

    sum = Sum(i)(Sum(j)(ket_i * ket_j.dag()))
    ful = OperatorIndexedSum(
        ket_i * ket_j.dag(),
        IndexOverFockSpace(i, hs0), IndexOverFockSpace(j, hs0))
    assert sum == ful

    #sum = Sum(i)(Sum(j)(ket_i.dag() * ket_j)) # TODO
    #assert sum == ful

    # TODO: sum over A_i


@pytest.fixture
def MyScalarFunc():

    class MyScalarDerivative(QuantumDerivative, Scalar):
        pass

    class ScalarFunc(ScalarExpression):

        def __init__(self, name, *sym_args):
            self._name = name
            self._sym_args = sym_args
            super().__init__(name, *sym_args)

        def _adjoint(self):
            return self

        @property
        def args(self):
            return (self._name, *self._sym_args)

        def _diff(self, sym):
            return MyScalarDerivative(self, derivs={sym: 1})

    return ScalarFunc


def test_quantum_derivative(MyScalarFunc):
    """Test the basic properties of a QuantumDerivative"""
    s, t, x = symbols('s, t, x', real=True)
    f = MyScalarFunc("f", s, t)
    assert f.diff(x) == Zero
    fdot = f.diff(s, n=2).diff(t)
    with pytest.raises(ValueError):
        fdot.__class__(f, derivs={t: 0})
    with pytest.raises(TypeError):
        f.diff(2)
    with pytest.raises(TypeError):
        fdot.__class__(f, derivs={2: 1})
    assert isinstance(fdot, QuantumDerivative)
    assert fdot.kwargs == OrderedDict(
        [('derivs', ((s, 2), (t, 1))), ('vals', None)])
    assert fdot.minimal_kwargs == {'derivs': ((s, 2), (t, 1))}
    assert fdot.derivs == {s: 2, t: 1}
    assert isinstance(fdot.derivs, OrderedDict)
    assert fdot.syms == {s, t}
    assert fdot.vals == dict()
    assert isinstance(fdot.vals, OrderedDict)
    assert fdot.free_symbols == set([t, s])
    assert len(fdot.bound_symbols) == 0
    assert fdot.n == 3
    assert fdot.adjoint() == fdot


def test_quantum_derivative_evaluated(MyScalarFunc):
    """Test the basic properties of a QuantumDerivative, evaluated at a
    point"""
    s, t, t0, x = symbols('s, t, t_0, x', real=True)
    f = MyScalarFunc("f", s, t)
    fdot = f.diff(s, n=2).diff(t)
    fdot = fdot.evaluate_at({t: t0})
    D = fdot.__class__
    with pytest.raises(ValueError):
        fdot.evaluate_at({x: t0})
    assert fdot == D(f, derivs={s: 2, t: 1}, vals={t: t0})
    assert fdot == D.create(f, derivs={s: 2, t: 1}, vals={t: t0})
    assert fdot.kwargs == OrderedDict(
        [('derivs', ((s, 2), (t, 1))), ('vals', ((t, t0), ))])
    assert fdot.minimal_kwargs == fdot.kwargs
    assert fdot.derivs == {s: 2, t: 1}
    assert fdot.syms == {s, t}
    assert fdot.vals == {t: t0}
    assert isinstance(fdot.vals, OrderedDict)
    assert fdot.free_symbols == set([s, t0])
    assert len(fdot.bound_symbols) == 1
    assert fdot.bound_symbols == set([t, ])
    assert fdot.all_symbols == set([s, t, t0])
    assert fdot.n == 3
    assert fdot.adjoint() == fdot
    assert fdot.diff(t0) == D(fdot, derivs={t0: 1})
    assert fdot.diff(t) == Zero
    assert fdot._diff(t) == Zero
    assert fdot.diff(s) == D(f, derivs={s: 3, t: 1}, vals={t: t0})
    with pytest.raises(TypeError):
        fdot.diff(2)
    with pytest.raises(TypeError):
        fdot._diff(2)


def test_quantum_derivative_nonatomic_free_symbols(MyScalarFunc):
    """Test the fee_symbols of an evaluated derivative for non-atomic
    symbols"""
    s = IndexedBase('s')
    t = IndexedBase('t')
    i = IdxSym('i')
    j = IdxSym('j')
    t0 = symbols('t_0', real=True)
    f = MyScalarFunc("f", s[i], t[j])
    fdot = f.diff(s[i], n=2).diff(t[j]).evaluate_at({t[j]: t0})
    assert fdot == fdot.__class__(
        f, derivs={s[i]: 2, t[j]: 1}, vals={t[j]: t0})
    assert fdot.kwargs == OrderedDict(
        [('derivs', ((s[i], 2), (t[j], 1))), ('vals', ((t[j], t0), ))])
    assert fdot.derivs == {s[i]: 2, t[j]: 1}
    assert fdot.syms == {s[i], t[j]}
    assert fdot.vals == {t[j]: t0}
    assert fdot.free_symbols == set([s.args[0], i, t0])
    assert fdot.bound_symbols == set([t.args[0], j])
    assert fdot.all_symbols == set([s.args[0], t.args[0], t0, i, j])
    assert fdot.diff(s[i]).n == 4
    assert fdot.diff(t[j]) == Zero

    f = MyScalarFunc("f", s[i], t[j], j)
    fdot = f.diff(s[i], n=2).diff(t[j]).evaluate_at({t[j]: t0})
    assert fdot.free_symbols == set([s.args[0], i, j, t0])
    assert fdot.bound_symbols == set([t.args[0], j])


def test_abstract_taylor_series(MyScalarFunc):
    """Test a series expansion that is the abstract Taylor series only"""
    s = IndexedBase('s')
    t = IndexedBase('t')
    i = IdxSym('i')
    j = IdxSym('j')
    t0 = symbols('t_0', real=True)
    f = MyScalarFunc("f", s[i], t[j])

    series = f.series_expand(t[j], about=0, order=3)
    assert isinstance(series[0], MyScalarFunc)
    assert isinstance(series[1], QuantumDerivative)
    assert isinstance(series[2], ScalarTimes)
    D = series[1].__class__
    assert series[0] == MyScalarFunc("f", s[i], 0)
    assert series[1] == D(f, derivs={t[j]: 1}, vals={t[j]: 0})
    assert series[2] == D(f, derivs={t[j]: 2}, vals={t[j]: 0}) / 2
    assert series[3] == D(f, derivs={t[j]: 3}, vals={t[j]: 0}) / 6

    series = f.series_expand(t[j], about=t0, order=3)
    assert series[0] == MyScalarFunc("f", s[i], t0)
    assert series[1] == D(f, derivs={t[j]: 1}, vals={t[j]: t0})
    assert series[2] == D(f, derivs={t[j]: 2}, vals={t[j]: t0}) / 2
    assert series[3] == D(f, derivs={t[j]: 3}, vals={t[j]: t0}) / 6


def test_quantum_symbols_with_symargs():
    """Test properties and behavior of symbols with scalar arguments,
    through the example of an OperatorSymbol"""
    t = IndexedBase('t')
    i = IdxSym('i')
    j = IdxSym('j')
    alpha, beta = symbols('alpha, beta')
    A = OperatorSymbol("A", t[i], (alpha + 1)**2, hs=0)
    assert A.label == 'A'
    assert len(A.args) == 3
    assert A.kwargs == {'hs': LocalSpace('0')}
    assert A._get_instance_key(A.args, A.kwargs) == (
        OperatorSymbol, 'A', t[i], (alpha + 1)**2, ('hs', A.space))
    A_beta = OperatorSymbol("A", beta, (alpha + 1)**2, hs=0)
    assert A != A_beta
    assert A.substitute({t[i]: beta}) == A_beta
    half = sympy.sympify(1) / 2
    assert A.sym_args == (t[i], (alpha + 1)**2)
    assert A.free_symbols == {symbols('t'), i, alpha}
    assert len(A.bound_symbols) == 0
    assert A.simplify_scalar(sympy.expand) == OperatorSymbol(
        "A", t[i], alpha**2 + 2*alpha + 1, hs=0)
    assert A.diff(beta) == ZeroOperator
    assert A.diff(t[j]) == ZeroOperator
    assert OperatorSymbol("A", t[i], i, j, hs=0).diff(t[j]) == ZeroOperator
    assert A.diff(alpha) == OperatorDerivative(A, derivs=((alpha, 1),))
    assert A.expand() == A
    series = A.series_expand(t[i], about=beta, order=2)
    assert len(series) == 3
    assert series[0] == OperatorSymbol("A", beta, (alpha + 1)**2, hs=0)
    assert series[2] == half * OperatorDerivative(
        A, derivs=((t[i], 2),), vals=((t[i], beta),))


def test_quantum_symbols_with_indexedhs():
    """Test the fee_symbols method for objects that have a Hilbert space with a
    sybmolic label, for the example of an OperatorSymbol"""
    i, j = symbols('i, j', cls=IdxSym)
    hs_i = LocalSpace(StrLabel(i))
    hs_j = LocalSpace(StrLabel(j))
    A = OperatorSymbol("A", hs=hs_i*hs_j)
    assert A.free_symbols == {i, j}
    expr = (Create(hs=hs_i)*Destroy(hs=hs_i))
    assert expr.free_symbols == {i, }
