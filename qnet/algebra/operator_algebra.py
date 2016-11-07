# coding=utf-8
# This file is part of QNET.
#
# QNET is free software: you can redistribute it and/or modify
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
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

r"""
Operator Algebra
================

This module features classes and functions to define and manipulate symbolic
Operator expressions.  For more details see :ref:`operator_algebra`.

For a list of all properties and methods of an operator object, see the
documentation for the basic :py:class:`Operator` class.
"""
from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import defaultdict
from itertools import product as cartesian_product

from qnet.algebra.abstract_algebra import (
        tex, singleton, Expression, Operation, assoc, orderby,
        filter_neutral, match_replace_binary, match_replace,
        set_union, KeyTuple, substitute, CannotSimplify)
from qnet.algebra.hilbert_space_algebra import (
        TrivialSpace, HilbertSpace, LocalSpace, ProductSpace, BasisNotSetError,
        FullSpace)
from qnet.algebra.pattern_matching import wc, pattern_head, pattern


from sympy import (
        exp, log, cos, sin, cosh, sinh, tan, cot, acos, asin, acosh,
        atan, atan2, atanh, sqrt, factorial, pi, I, sympify,
        Matrix as SympyMatrix, Basic as SympyBasic, symbols, Add,
        series as sympy_series)
from sympy.printing import latex as sympy_latex

from numpy import (
        array as np_array, shape as np_shape, hstack as np_hstack,
        vstack as np_vstack, diag as np_diag, ones as np_ones,
        conjugate as np_conjugate, zeros as np_zeros, ndarray, arange,
        cos as np_cos, sin as np_sin, eye as np_eye, argwhere,
        int64, complex128, float64)
from qnet.algebra.permutations import check_permutation

sympyOne = sympify(1)

identity_str = '𝟙'
identity_tex = r'\openone'
dagger_str = '⁺'
dagger_tex = r'^{\dagger}'
pseudo_dagger_str = '⁺⁺'
pseudo_dagger_tex = r'^{+}'
projector_str = 'Π'
projector_tex = r'\Pi'
sigma_str = 'σ'
sigma_tex = r'\sigma'


def op_tex(name, hs_name=None, subscript=None, dagger=False, args=None):
    """Return a tex representation of an operator"""
    res = "\hat{"+identifier_to_tex(name)+"}"
    total_subscript = ''
    if (subscript is not None) and (hs_name is None):
        total_subscript = subscript
    elif (subscript is None) and (hs_name is not None):
        total_subscript = "(" + hs_name + ")"
    elif (subscript is not None) and (hs_name is not None):
        total_subscript = "%s,(%s)" % (subscript, hs_name)
    if len(total_subscript) > 0:
        if len(total_subscript) > 1:
            res += '_{%s}' % total_subscript
        else:
            res += '_%s' % total_subscript
    if dagger:
        res += dagger_str
    if args is not None and len(args) > 0:
        str_args = [str(arg) for arg in args]
        res += '(' + ", ".join(str_args) + ')'
    return res


projector_str = 'Π'
projector_tex = r'\Pi'
sigma_str = 'σ'
sigma_tex = r'\sigma'


def op_tex(name, hs_name=None, subscript=None, dagger=False, args=None):
    """Return a tex representation of an operator"""
    res = "\hat{"+identifier_to_tex(name)+"}"
    total_subscript = ''
    if (subscript is not None) and (hs_name is None):
        total_subscript = subscript
    elif (subscript is None) and (hs_name is not None):
        total_subscript = "(" + hs_name + ")"
    elif (subscript is not None) and (hs_name is not None):
        total_subscript = "%s,(%s)" % (subscript, hs_name)
    if len(total_subscript) > 0:
        if len(total_subscript) > 1:
            res += '_{%s}' % total_subscript
        else:
            res += '_%s' % total_subscript
    if dagger:
        res += dagger_tex
    if args is not None and len(args) > 0:
        tex_args = [identifier_to_tex(arg) for arg in args]
        res += '(' + ", ".join(tex_args) + ')'
    return res


unicode_subscript_mapping = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
        '7': '₇', '8': '₈', '9': '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋',
        '=': '₌', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ', 'k': 'ₖ',
        'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
        }


def unicode_subscript(subscript):
    """Try to render a subscript string in unicode, fall back on ascii if this
    is not possible"""
    unicode_letters = []
    for letter in subscript:
        try:
            unicode_letters.append(unicode_subscript_mapping[letter])
        except KeyError:
            if len(subscript) > 1:
                return '_{%s}' % subscript
            else:
                return '_%s' % subscript
    return ''.join(unicode_letters)


def op_str(name, hs_name=None, subscript=None, dagger=False, args=None):
    """Return a tex representation of an operator"""
    res = name
    if dagger:
        res += dagger_str
    total_subscript = ''
    if (subscript is not None) and (hs_name is None):
        total_subscript = subscript
    elif (subscript is None) and (hs_name is not None):
        total_subscript = "(" + hs_name + ")"
    elif (subscript is not None) and (hs_name is not None):
        total_subscript = "%s,(%s)" % (subscript, hs_name)
    res += unicode_subscript(total_subscript)
    if args is not None and len(args) > 0:
        str_args = [str(arg) for arg in args]
        res += '(' + ", ".join(str_args) + ')'
    return res


def adjoint(obj):
    """Return the adjoint of an obj."""
    try:
        return obj.adjoint()
    except AttributeError:
        return obj.conjugate()


class Operator(metaclass=ABCMeta):
    """The basic operator class, which fixes the abstract interface of operator
    objects and where possible also defines the default behavior under
    operations.  Any operator contains an associated HilbertSpace object,
    on which it is taken to act non-trivially.
    """

    # which data types may serve as scalar coefficients
    scalar_types = (int, float, complex, SympyBasic, int64, complex128,
                    float64)

    @abstractproperty
    def space(self):
        """The HilbertSpace on which the operator acts non-trivially"""
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """The Hermitian adjoint of the operator."""
        return Adjoint.create(self)

    conjugate = dag = adjoint

    def pseudo_inverse(self):
        """The pseudo-Inverse of the Operator, i.e., it inverts the operator on
        the orthogonal complement of its nullspace"""
        return PseudoInverse.create(self)

    def expand(self):
        """Expand out distributively all products of sums. Note that this does
        not expand out sums of scalar coefficients.

        :return: A fully expanded sum of operators.
        :rtype: Operator
        """
        return self._expand()

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

    def simplify_scalar(self):
        """Simplify all scalar coefficients within the Operator expression.

        :return: The simplified expression.
        :rtype: Operator
        """
        return self._simplify_scalar()

    def _simplify_scalar(self):
        return self

    def diff(self, sym, n=1, expand_simplify=True):
        """Differentiate by scalar parameter sym.

        :param (sympy.Symbol) sym: What to differentiate by.
        :param (int) n: How often to differentiate
        :param (bool) expand_simplify: Whether to simplify the result.
        :return (Operator): The n-th derivative.
        """
        expr = self
        for k in range(n):
            expr = expr._diff(sym)
        if expand_simplify:
            expr = expr.expand().simplify_scalar()
        return expr

    def _diff(self, sym):
        return ZeroOperator

    def series_expand(self, param, about, order):
        """Expand the operator expression as a truncated power series in a
        scalar parameter.

        :param param: Expansion parameter.
        :type param: sympy.core.symbol.Symbol
        :param about: Point about which to expand.
        :type about:  Any one of Operator.scalar_types
        :param order: Maximum order of expansion.
        :type order: int >= 0
        :return: tuple of length (order+1), where the entries are the expansion coefficients.
        :rtype: tuple of Operator
        """
        return self._series_expand(param, about, order)

    @abstractmethod
    def _series_expand(self, param, about, order):
        raise NotImplementedError(self.__class__.__name__)

    def __add__(self, other):
        if isinstance(other, Operator.scalar_types):
            return OperatorPlus.create(self, other * IdentityOperator)
        elif isinstance(other, Operator):
            return OperatorPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Operator.scalar_types):
            return ScalarTimesOperator.create(other, self)
        elif isinstance(other, Operator):
            return OperatorTimes.create(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator.scalar_types):
            return ScalarTimesOperator.create(other, self)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented

    __truediv__ = __div__

    def __pow__(self, other):
        if isinstance(other, int):
            return OperatorTimes.create(*[self for __ in range(other)])
        else:
            return NotImplemented


def space(obj):
    """Gives the associated HilbertSpace with an object. Also works for
    :py:attr:`Operator.scalar_types`.

    :type obj: Operator or Operator.scalar_types
    :rtype: HilbertSpace
    """
    try:
        return obj.space
    except AttributeError:
        if isinstance(obj, Operator.scalar_types):
            return TrivialSpace
        raise ValueError(str(obj))


class OperatorSymbol(Operator, Expression):
    """Symbolic operator, parametrized by an identifier string and an
    associated Hilbert space.

        ``OperatorSymbol(name, hs)``

    :param name: Symbol identifier
    :type name: str
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace
    """
    def __init__(self, name, hs):
        self.name = name
        if isinstance(hs, (str, int)):
            self._hs = LocalSpace(hs)
        elif isinstance(hs, tuple):
            self._hs = ProductSpace.create(*[LocalSpace(h) for h in hs])
        else:
            self._hs = hs

    @property
    def args(self):
        return self.name, self._hs

    def __str__(self):
        return op_str(self.name)

    def tex(self):
        return op_tex(self.name)

    @property
    def space(self):
        return self._hs

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def all_symbols(self):
        return {self}


@singleton
class IdentityOperator(Operator, Expression):
    """``IdentityOperator`` constant (singleton) object."""

    @property
    def space(self):
        """TrivialSpace"""
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def adjoint(self):
        return self

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def pseudo_inverse(self):
        return self

    def tex(self):
        return identity_tex

    def __str__(self):
        return identity_str

    def __eq__(self, other):
        return self is other  # or other == 1

    def all_symbols(self):
        return set(())


II = IdentityOperator


@singleton
class ZeroOperator(Operator, Expression):
    """``ZeroOperator`` constant (singleton) object."""

    @property
    def space(self):
        """TrivialSpace"""
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def adjoint(self):
        return self

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def pseudo_inverse(self):
        return self

    def tex(self):
        return "0"

    def __eq__(self, other):
        return self is other or other == 0

    def __str__(self):
        return "0"

    def all_symbols(self):
        return set(())


def implied_local_space(*, arg_index=None, keys=None):
    """Return a simplification that converts the positional argument
    `arg_index` from (str, int) to LocalSpace, as well as any keyword argument
    with one of the given keys"""

    def args_to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected args to LocalSpace"""
        if isinstance(args[arg_index], LocalSpace):
            new_args = args
        else:
            if isinstance(args[arg_index], (int, str)):
                hs = LocalSpace(args[arg_index])
            else:
                hs = args[arg_index]
                assert isinstance(hs, HilbertSpace)
            new_args = (tuple(args[:arg_index]) + (hs, ) +
                        tuple(args[arg_index+1:]))
        return new_args, kwargs

    def kwargs_to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected kwargs to LocalSpace"""
        if all([isinstance(kwargs[key], LocalSpace) for key in keys]):
            new_kwargs = kwargs
        else:
            new_kwargs = {}
            for key, val in kwargs.items():
                if key in keys:
                    if isinstance(val, (int, str)):
                        val = LocalSpace(val)
                    assert isinstance(val, HilbertSpace)
                new_kwargs[key] = val
        return args, new_kwargs

    def to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected args and kwargs to LocalSpace"""
        new_args, __ = args_to_local_space(args, kwargs, arg_index)
        __, new_kwargs = kwargs_to_local_space(args, kwargs, keys)
        return new_args, new_kwargs

    if (arg_index is not None) and (keys is None):
        return args_to_local_space
    elif (arg_index is None) and (keys is not None):
        return kwargs_to_local_space
    elif (arg_index is not None) and (keys is not None):
        return to_local_space
    else:
        raise ValueError("must give at least one of arg_index and keys")


class LocalOperator(Operator, Expression, metaclass=ABCMeta):
    """Base class for all kinds of operators that act *locally*,
    i.e. only on a single degree of freedom."""

    _simplifications = [implied_local_space(arg_index=0), ]

    def __init__(self, hs):
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        assert isinstance(hs, LocalSpace)
        self._hs = hs

    @property
    def space(self):
        return self._hs

    @property
    def args(self):
        return (self._hs, )

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def all_symbols(self):
        return set()


class Create(LocalOperator):
    """``Create(space)`` yields a bosonic creation operator acting on a
    particular local space/degree of freedom.
    Its adjoint is

        >>> print(Create(1).adjoint())
        a₍₁₎

    and it obeys the bosonic commutation relation

        >>> Destroy(1) * Create(1) - Create(1) * Destroy(1)
        IdentityOperator
        >>> Destroy(1) * Create(2) - Create(2) * Destroy(1)
        ZeroOperator

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    """
    def tex(self):
        hs_name = self.space.name
        return op_tex('a', hs_name=hs_name, dagger=True)

    def __str__(self):
        hs_name = self.space.name
        return op_str('a', hs_name=hs_name, dagger=True)


class Destroy(LocalOperator):
    """``Destroy(space)`` yields a bosonic annihilation operator acting on a
    particular local space/degree of freedom.
    Its adjoint is

        >>> print(Destroy(1).adjoint())
        a⁺₍₁₎

    and it obeys the bosonic commutation relation

        >>> Destroy(1) * Create(1) - Create(1) * Destroy(1)
        IdentityOperator
        >>> Destroy(1) * Create(2) - Create(2) * Destroy(1)
        ZeroOperator

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    """
    def tex(self):
        hs_name = self.space.name
        return op_tex('a', hs_name=hs_name, dagger=False)

    def __str__(self):
        hs_name = self.space.name
        return op_str('a', hs_name=hs_name, dagger=False)


class Jz(LocalOperator):
    """``Jz(space)`` yields the z-component of a general spin operator acting
    on a particular local space/degree of freedom with well defined spin
    quantum number J.  It is Hermitian

        >>> print(Jz(1).adjoint())
        J_{z,(1)}

    Jz, Jplus and Jminus satisfy that angular momentum commutator algebra

        >>> print((Jz(1) * Jplus(1) - Jplus(1)*Jz(1)).expand())
        J_{+,(1)}

        >>> print((Jz(1) * Jminus(1) - Jminus(1)*Jz(1)).expand())
         -J_{-,(1)}

        >>> print((Jplus(1) * Jminus(1) - Jminus(1)*Jplus(1)).expand())
        2 * J_{z,(1)}

    where Jplus = Jx + i * Jy, Jminux= Jx - i * Jy.
    """
    def tex(self):
        hs_name = self.space.name
        return op_tex('J', hs_name=hs_name, subscript='z', dagger=False)

    def __str__(self):
        hs_name = self.space.name
        return op_str('J', hs_name=hs_name, subscript='z', dagger=False)


class Jplus(LocalOperator):
    """ ``Jplus(space)`` yields the J_+ raising ladder operator of a general
    spin operator acting on a particular local space/degree of freedom with
    well defined spin quantum number J.  It's adjoint is the lowering operator

        >>> print(Jplus(1).adjoint())
        J_{-,(1)}

    Jz, Jplus and Jminus satisfy that angular momentum commutator algebra

        >>> print((Jz(1) * Jplus(1) - Jplus(1)*Jz(1)).expand())
        J_{+,(1)}

        >>> print((Jz(1) * Jminus(1) - Jminus(1)*Jz(1)).expand())
         -J_{-,(1)}

        >>> print((Jplus(1) * Jminus(1) - Jminus(1)*Jplus(1)).expand())
        2 * J_{z,(1)}

    where Jplus = Jx + i * Jy, Jminux= Jx - i * Jy.
    """
    def tex(self):
        hs_name = self.space.name
        return op_tex('J', hs_name=hs_name, subscript='+', dagger=False)

    def __str__(self):
        hs_name = self.space.name
        return op_str('J', hs_name=hs_name, subscript='+', dagger=False)


class Jminus(LocalOperator):
    """``Jminus(space)`` yields the J_- lowering ladder operator of a general
    spin operator acting on a particular local space/degree of freedom with
    well defined spin quantum number J.  It's adjoint is the raising operator

        >>> print(Jminus(1).adjoint())
        J_{+,(1)}

    Jz, Jplus and Jminus satisfy that angular momentum commutator algebra

        >>> print((Jz(1) * Jplus(1) - Jplus(1)*Jz(1)).expand())
        J_{+,(1)}

        >>> print((Jz(1) * Jminus(1) - Jminus(1)*Jz(1)).expand())
         -J_{-,(1)}

        >>> print((Jplus(1) * Jminus(1) - Jminus(1)*Jplus(1)).expand())
        2 * J_{z,(1)}

    where Jplus = Jx + i * Jy, Jminux= Jx - i * Jy.
    """
    def tex(self):
        hs_name = self.space.name
        return op_tex('J', hs_name=hs_name, subscript='-', dagger=False)

    def __str__(self):
        hs_name = self.space.name
        return op_str('J', hs_name=hs_name, subscript='-', dagger=False)


def simplify_scalar(s):
    """Simplify all occurences of scalar expressions in s

    :param s: The expression to simplify.
    :type s: Expression or SympyBasic
    :return: The simplified version.
    :rtype: Expression or SympyBasic
    """
    try:
        return s.simplify_scalar()
    except AttributeError:
        pass
    if isinstance(s, SympyBasic):
        return s.simplify()
    return s


def scalar_free_symbols(*operands):
    """Return all free symbols from any symbolic operand"""
    if len(operands) > 1:
        return set_union([scalar_free_symbols(o) for o in operands])
    elif len(operands) < 1:
        return set()
    else:  # len(operands) == 1
        o, = operands
        if isinstance(o, SympyBasic):
            return set(o.free_symbols)
    return set()


class Phase(LocalOperator):
    r"""The unitary Phase operator acting on a particular local space/degree of
    freedom:

    .. math::

        P_{\rm s}(\phi):= \exp\left(i \phi a_{\rm s}^\dagger a_{\rm s}\right)

    where :math:`a_{\rm s}` is the annihilation operator acting on the local
    space s.
    Use as:

        ``Phase.create(hs, phi)``

    :param hs: Associated local Hilbert space.
    :type hs: LocalSpace or str
    :param phi: Displacement amplitude.
    :type phi: Any from `Operator.scalar_types`
    """
    _rules = []  # see end of module
    _simplifications = [implied_local_space(arg_index=0), match_replace]

    def __init__(self, hs, phi):
        self.phi = phi
        super().__init__(hs)

    @property
    def args(self):
        return self._hs, self.phi

    def _diff(self, sym):
        raise NotImplementedError()

    def adjoint(self):
        return Phase(self.space, -self.phi.conjugate())

    def pseudo_inverse(self):
        return Phase(self.space, -self.phi)

    def tex(self):
        hs_name = self.space.name
        return op_tex('P', hs_name=hs_name, args=[self.phi, ])

    def __str__(self):
        hs_name = self.space.name
        return op_str('P', hs_name=hs_name, args=[self.phi, ])

    def _simplify_scalar(self):
        return Phase(self.space, simplify_scalar(self.phi))

    def all_symbols(self):
        return scalar_free_symbols(self.space)


class Displace(LocalOperator):
    r"""Unitary coherent displacement operator

    .. math::

        D_{\rm s}(\alpha) := \exp\left({\alpha a_{\rm s}^\dagger - \alpha^* a_{\rm s}}\right)

    where :math:`a_{\rm s}` is the annihilation operator acting on the local
    space s.
    Use as:

        ``Displace.create(space, alpha)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param alpha: Displacement amplitude.
    :type alpha: Any from `Operator.scalar_types`
    """
    _rules = []  # see end of module
    _simplifications = [implied_local_space(arg_index=0), match_replace]

    def __init__(self, hs, alpha):
        self.alpha = alpha
        super().__init__(hs)

    @property
    def args(self):
        return self._hs, self.alpha

    def _diff(self, sym):
        raise NotImplementedError()

    def adjoint(self):
        return Displace(self.space, -self.alpha)

    pseudo_inverse = adjoint

    def tex(self):
        hs_name = self.space.name
        return op_tex('D', hs_name=hs_name, args=[self.alpha, ])

    def __str__(self):
        hs_name = self.space.name
        return op_str('D', hs_name=hs_name, args=[self.alpha, ])

    def _simplify_scalar(self):
        return Displace(self.space, simplify_scalar(self.alpha))

    def all_symbols(self):
        return scalar_free_symbols(self.space)


class Squeeze(LocalOperator):
    r"""A unitary Squeezing operator acting on a particular local space/degree
    of freedom:

    .. math::

        S_{\rm s}(\eta) := \exp {\left( \frac{\eta}{2} {a_{\rm s}^\dagger}^2 - \frac{\eta^*}{2} {a_{\rm s}}^2 \right)}

    where :math:`a_{\rm s}` is the annihilation operator acting on the local space s.
    Use as:

        ``Squeeze(space, eta)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param eta: Squeeze parameter.
    :type eta: Any from `Operator.scalar_types`
    """
    _rules = []  # see end of module
    _simplifications = [implied_local_space(arg_index=0), match_replace]

    def __init__(self, hs, eta):
        self.eta = eta
        super().__init__(hs)

    @property
    def args(self):
        return self._hs, self.eta

    def _diff(self, sym):
        raise NotImplementedError()

    def adjoint(self):
        return Squeeze(self.space, -self.eta)

    pseudo_inverse = adjoint

    def tex(self):
        hs_name = self.space.name
        return op_tex('S', hs_name=hs_name, args=[self.eta, ])

    def __str__(self):
        hs_name = self.space.name
        return op_str('S', hs_name=hs_name, args=[self.eta, ])

    def _simplify_scalar(self):
        return Squeeze(self.space, simplify_scalar(self.eta))

    def all_symbols(self):
        return scalar_free_symbols(self.space)


class LocalSigma(LocalOperator):
    r"""A local level flip operator operator acting on a particular local
    space/degree of freedom.

    .. math::

        \sigma_{jk}^{\rm s} := \left| j\right\rangle_{\rm s} \left \langle k \right |_{\rm s}

    Use as:

        ``LocalSigma(space, j, k)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    :param j: State label j.
    :type j: int or str
    :param k: State label k.
    :type k: int or str
    """
    def __init__(self, hs, j, k):
        self.j = j
        self.k = k
        super().__init__(hs)

    def tex(self):
        j, k = self.j, self.k
        hs_name = self.space.name
        if k == j:
            return op_tex(projector_tex, hs_name=hs_name, subscript=j)
        else:
            return op_tex(sigma_tex, hs_name=hs_name,
                          subscript='{%s,%s}' % (j, k))
        j, k = self.j, self.k

    @property
    def args(self):
        return self._hs, self.j, self.k

    def __str__(self):
        j, k = self.j, self.k
        hs_name = self.space.name
        if k == j:
            return op_str(projector_str, hs_name=hs_name, subscript=j)
        else:
            return op_str(sigma_str, hs_name=hs_name,
                          subscript='%s,%s' % (j, k))


LocalProjector = lambda hs, state: LocalSigma.create(hs, state, state)


def X(local_space, states=("h", "g")):
    r"""Pauli-type X-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or str
    :return: Local X-operator.
    :rtype: Operator
    """
    h, g = states
    return LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h)


def Y(local_space, states=("h", "g")):
    r""" Pauli-type Y-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or str
    :return: Local Y-operator.
    :rtype: Operator
    """
    h, g = states
    return I * (-LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h))


def Z(local_space, states=("h", "g")):
    r"""Pauli-type Z-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or str
    :return: Local Z-operator.
    :rtype: Operator
    """
    h, g = states
    return LocalProjector(local_space, h) - LocalProjector(local_space, g)


class OperatorOperation(Operator, Operation):
    """Base class for Operations acting only on Operator arguments."""
    signature = (Operator, '*'), {}

    @property
    def space(self):
        op_spaces = [o.space for o in self.operands]
        return ProductSpace.create(*op_spaces)

    def _simplify_scalar(self):
        return self.__class__.create(*[o.simplify_scalar()
                                       for o in self.operands])


class OperatorPlus(OperatorOperation):
    """A sum of Operators

        ``OperatorPlus(*summands)``

    :param summands: Operator summands.
    :type summands: Operator
    """
    neutral_element = ZeroOperator
    _binary_rules = []
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary,
                        filter_neutral]

    @classmethod
    def order_key(cls, a):

        if isinstance(a, ScalarTimesOperator):
            c = a.coeff
            if isinstance(c, SympyBasic):
                c = float('inf')
            return KeyTuple((Expression.order_key(a.term), c))
        return KeyTuple((Expression.order_key(a), 1))

    def _expand(self):
        summands = [o.expand() for o in self.operands]
        return OperatorPlus.create(*summands)

    def _series_expand(self, param, about, order):
        tuples = (o.series_expand(param, about, order) for o in self.operands)
        res = (OperatorPlus.create(*tels) for tels in zip(*tuples))
        return res

    def _diff(self, sym):
        return sum([o._diff(sym) for o in self.operands], ZeroOperator)

    @staticmethod
    def _conditional_wrap(op, istex=False):
        if istex:
            if isinstance(op, OperatorPlus):
                return r"\left( " + tex(op) + r"\right)"
            else:
                return tex(op)
        else:
            if isinstance(op, OperatorPlus):
                return r"( " + str(op) + " )"
            else:
                return str(op)

    def tex(self):
        "TeX representation of operator sum"""
        _cw = OperatorPlus._conditional_wrap
        ret = _cw(self.operands[0].tex(), istex=True)

        for o in self.operands[1:]:
            if (isinstance(o, ScalarTimesOperator) and
                    ScalarTimesOperator.has_minus_prefactor(o.coeff)):
                ret += " - " + _cw(-o, istex=True)
            else:
                ret += " + " + _cw(o, istex=True)
        return ret

    def __str__(self):
        _cw = OperatorPlus._conditional_wrap
        ret = _cw(self.operands[0])

        for o in self.operands[1:]:
            if (isinstance(o, ScalarTimesOperator) and
                    ScalarTimesOperator.has_minus_prefactor(o.coeff)):
                ret += " - " + _cw(-o)
            else:
                ret += " + " + _cw(o)
        return ret


def _coeff_term(op):
    if isinstance(op, ScalarTimesOperator):
        return op.coeff, op.term
    else:
        return 1, op


def factor_coeff(cls, ops, kwargs):
    """Factor out coefficients of all factors."""
    coeffs, nops = zip(*map(_coeff_term, ops))
    coeff = 1
    for c in coeffs:
        coeff *= c
    if coeff == 1:
        return nops, coeffs
    else:
        return coeff * cls.create(*nops, **kwargs)


class OperatorOrderKey():
    """Auxiliary class that generates the correct pseudo-order relation for
    operator products.  Only operators acting on different Hilbert spaces
    are commuted to achieve the order specified in the full HilbertSpace.
    I.e., ``sorted(factors, key=OperatorOrderKey)`` achieves this ordering.
    """

    def __init__(self, op):
        s = op.space
        self.op = op
        self.full = False
        self.trivial = False
        self.local_spaces = set()
        if isinstance(s, LocalSpace):
            self.local_spaces = {s.args, }
        elif s is TrivialSpace:
            self.trivial = True
        elif s is FullSpace:
            self.full = True
        else:
            assert isinstance(s, ProductSpace)
            self.local_spaces = {s.args for s in s.operands}

    def __lt__(self, other):
        if self.trivial and other.trivial:
            return (Expression.order_key(self.op) <
                    Expression.order_key(other.op))
        if self.full or len(self.local_spaces & other.local_spaces):
            return False
        return tuple(self.local_spaces) < tuple(other.local_spaces)

    def __gt__(self, other):
        if self.trivial and other.trivial:
            return (Expression.order_key(self.op) >
                    Expression.order_key(other.op))

        if self.full or len(self.local_spaces & other.local_spaces):
            return False

        return tuple(self.local_spaces) > tuple(other.local_spaces)

    def __eq__(self, other):
        if self.trivial and other.trivial:
            return (Expression.order_key(self.op) ==
                    Expression.order_key(other.op))

        return self.full or len(self.local_spaces & other.local_spaces) > 0


class OperatorTimes(OperatorOperation):
    """A product of Operators that serves both as a product within a Hilbert
    space as well as a tensor product.

        ``OperatorTimes(*factors)``

    :param factors: Operator factors.
    :type factors: Operator
    """

    neutral_element = IdentityOperator
    _binary_rules = []  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary,
                        filter_neutral]

    order_key = OperatorOrderKey

    def factor_for_space(self, spc):
        if spc == TrivialSpace:
            ops_on_spc = [o for o in self.operands
                          if o.space is TrivialSpace]
            ops_not_on_spc = [o for o in self.operands
                              if o.space > TrivialSpace]
        else:
            ops_on_spc = [o for o in self.operands
                          if (o.space & spc) > TrivialSpace]
            ops_not_on_spc = [o for o in self.operands
                              if (o.space & spc) is TrivialSpace]
        return (OperatorTimes.create(*ops_on_spc),
                OperatorTimes.create(*ops_not_on_spc))

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, OperatorPlus) else (eo,)
                        for eo in eops]
        # iterate over a cartesian product of all factor summands, form product
        # of each tuple and sum over result
        summands = []
        for combo in cartesian_product(*eopssummands):
            summand = OperatorTimes.create(*combo)
            summands.append(summand)
        ret = OperatorPlus.create(*summands)
        if isinstance(ret, OperatorPlus):
            return ret.expand()
        else:
            return ret

    def _series_expand(self, param, about, order):
        assert len(self.operands) > 1
        cfirst = self.operands[0].series_expand(param, about, order)
        crest = OperatorTimes.create(*self.operands[1:]).series_expand(
                    param, about, order)
        res = []
        for n in range(order + 1):
            summands = [cfirst[k] * crest[n - k] for k in range(n + 1)]
            res.append(OperatorPlus.create(*summands))
        return tuple(res)

    def _diff(self, sym):
        assert len(self.operands) > 1
        first = self.operands[0]
        rest = OperatorTimes.create(*self.operands[1:])
        return first._diff(sym) * rest + first * rest._diff(sym)


    def tex(self):
        """TeX representation of operator product"""
        ret = ""
        for o in self.operands:
            if isinstance(o, OperatorPlus):
                ret += r" \left({}\right) ".format(tex(o))
            else:
                ret += " {}".format(tex(o))
        return ret.strip()

    def __str__(self):
        ret = ""
        for o in self.operands:
            if isinstance(o, OperatorPlus):
                ret += r" ({}) ".format(str(o))
            else:
                ret += " {}".format(str(o))
        return ret.strip()


class ScalarTimesOperator(Operator, Operation):
    """Multiply an operator by a scalar coefficient.

        ``ScalarTimesOperator(coefficient, term)``

    :param coefficient: Scalar coefficient.
    :type coefficient: Any of Operator.scalar_types
    :param term: The operator that is multiplied.
    :type term: Operator
    """
    signature = (Operator.scalar_types, Operator), {}
    _rules = []
    _simplifications = [match_replace, ]

    @staticmethod
    def has_minus_prefactor(c):
        """
        For a scalar object c, determine whether it is prepended by a "-" sign.
        """
        cs = str(c).strip()
        return cs[0] == "-"

    @property
    def space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def tex(self):
        """TeX representation of operator"""
        coeff, term = self.operands

        if ScalarTimesOperator.has_minus_prefactor(coeff):
            return " -" + (-self).tex()

        if isinstance(coeff, Add):
            cs = r" \left({}\right)".format(tex(coeff))
        else:
            cs = " {}".format(tex(coeff))

        if term is IdentityOperator:
            ct = ""
        elif isinstance(term, OperatorPlus):
            ct = r" \left({}\right)".format(term.tex())
        else:
            ct = r" {}".format(term.tex())
        return (cs + ct).strip()

    def __str__(self):
        coeff, term = self.operands
        if isinstance(coeff, Add):
            cs = r"({!s}) ".format(coeff)
        else:
            cs = " {!s} ".format(coeff)
        if ScalarTimesOperator.has_minus_prefactor(coeff):
            return " -" + str(-self)
        if term == IdentityOperator:
            ct = ""
        if isinstance(term, OperatorPlus):
            ct = r" ({!s})".format(term)
        else:
            ct = r" {!s}".format(term)
        return cs.strip() + ' * ' + ct.strip()

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, OperatorPlus):
            summands = [c * eto for eto in et.operands]
            return OperatorPlus.create(*summands)
        return c * et

    def _series_expand(self, param, about, order):
        te = self.term.series_expand(param, about, order)
        if isinstance(self.coeff, SympyBasic):
            if about != 0:
                c = self.coeff.subs({param: about + param})
            else:
                c = self.coeff
            try:
                ce = list(reversed(
                        sympy_series(c, x=param, x0=0, n=order + 1)
                        .as_poly(param).all_coeffs()))
            except AttributeError:
                ce = [c] + [0] * order

            if len(ce) < order + 1:
                ce += [0] * (order + 1 - len(ce))
            res = []
            for n in range(order + 1):
                summands = [ce[k] * te[n - k] for k in range(n + 1)]
                res.append(OperatorPlus.create(*summands))
            return tuple(res)
        else:
            return tuple(self.coeff * tek for tek in te)

    def _diff(self, sym):
        c, t = self.operands
        cd = c.diff(sym) if isinstance(c, SympyBasic) else 0
        return cd*t + c * t._diff(sym)

    def pseudo_inverse(self):
        c, t = self.operands
        return t.pseudo_inverse() / c

    def __eq__(self, other):
        if (self.term is IdentityOperator and
                isinstance(other, Operator.scalar_types)):
            return self.coeff == other
        return super().__eq__(other)

    # When overloading __eq__ it can mess up the hashability unless we
    # explicitly also overload  __hash__, very subtle python3 bug, first found
    # by David Richie.
    def __hash__(self):
        return super().__hash__()

    def _substitute(self, var_map):
        st = self.term.substitute(var_map)
        if isinstance(self.coeff, SympyBasic):
            svar_map = {k: v for k, v in var_map.items()
                        if not isinstance(k, Expression)}
            sc = self.coeff.subs(svar_map)
        else:
            sc = substitute(self.coeff, var_map)
        return sc * st

    def _simplify_scalar(self):
        coeff, term = self.operands
        return simplify_scalar(coeff) * term.simplify_scalar()

    def all_symbols(self):
        return scalar_free_symbols(self.coeff) | self.term.all_symbols()


greek_letter_strings = [
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi",
    "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]

greekToLatex = {
    "alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "delta": "Delta",
    "epsilon": "Epsilon", "varepsilon": "Epsilon", "zeta": "Zeta",
    "eta": "Eta", "theta": "Theta", "vartheta": "Theta", "iota": "Iota",
    "kappa": "Kappa", "lambda": "Lambda", "mu": "Mu", "nu": "Nu", "xi": "Xi",
    "pi": "Pi", "varpi": "Pi", "rho": "Rho", "varrho": "Rho", "sigma": "Sigma",
    "varsigma": "Sigma", "tau": "Tau", "upsilon": "Upsilon", "phi": "Phi",
    "varphi": "Phi", "chi": "Chi", "psi": "Psi", "omega": "Omega",
    "Gamma": "CapitalGamma", "Delta": "CapitalDelta", "Theta": "CapitalTheta",
    "Lambda": "CapitalLambda", "Xi": "CapitalXi", "Pi": "CapitalPi", "Sigma":
    "CapitalSigma", "Upsilon": "CapitalUpsilon", "Phi": "CapitalPhi",
    "Psi": "CapitalPsi", "Omega": "CapitalOmega"}

import re

_idtp = re.compile(r'(?!\\)({})(\b|_)'.format("|".join(greek_letter_strings)))


def identifier_to_tex(identifier):
    """If an identifier contains a greek symbol name as a separate word, (e.g.
    ``my_alpha_1`` contains ``alpha`` as a separate word, but ``alphaman``
    doesn't) add a backslash in front.

    :param identifier: The string to prepare for LaTeX printing
    :type identifier: str
    :returns: An improved version where greek letter symbols can be correctly rendered.
    :rtype: str
    """
    #    identifier = creduce(lambda a,b: "{%s_%s}" % (b, a), ["{%s}" % part for part in reversed(identifier.split("_"))])

    ret = _idtp.sub(r'{\\\1}\2', identifier)
    return ret


class Adjoint(OperatorOperation):
    """
    The symbolic Adjoint of an operator.

        ``Adjoint(op)``

    :param op: The operator to take the adjoint of.
    :type op: Operator
    """
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def operand(self):
        return self.operands[0]

    def _series_expand(self, param, about, order):
        ope = self.operand.series_expand(param, about, order)
        return tuple(adjoint(opet) for opet in ope)

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, OperatorPlus):
            summands = [adjoint(eoo) for eoo in eo.operands]
            return OperatorPlus.create(*summands)
        return eo.adjoint()

    def pseudo_inverse(self):
        return self.operand.pseudo_inverse().adjoint()

    def tex(self):
        return "{%s}%s" % (tex(self.operand), dagger_tex)

    def __str__(self):
        if isinstance(self.operand, OperatorSymbol):
            return str(self.operand) + dagger_str
        return "(%s)%s" % (self.operand, dagger_str)

    def _diff(self, sym):
        return Adjoint.create(self.operands[0]._diff(sym))

# for hilbert space dimensions less than or equal to this,
# compute numerically PseudoInverse and NullSpaceProjector representations
DENSE_DIMENSION_LIMIT = 1000


def delegate_to_method(mtd):
    def _delegate_to_method(cls, ops, kwargs):
        """Factor out coefficients of all factors."""
        try:
            op, = ops
            if isinstance(op. cls._delegate_to_method):
                return getattr(op, mtd)()
        except (ValueError, AttributeError):
            return ops, kwargs
    return _delegate_to_method


class PseudoInverse(OperatorOperation):
    r"""The symbolic pseudo-inverse :math:`X^+` of an operator :math:`X`. It is
    defined via the relationship

    .. math::

        X X^+ X =  X  \\
        X^+ X X^+ =  X^+  \\
        (X^+ X)^\dagger = X^+ X  \\
        (X X^+)^\dagger = X X^+

    Use as:

        ``PseudoInverse(X)``

    :param X: The operator to take the adjoint of.
    :type X: Operator
    """
    _rules = []  # see end of module
    _delegate_to_method = (ScalarTimesOperator, Squeeze, Displace,
                           ZeroOperator.__class__, IdentityOperator.__class__)
    _simplifications = [match_replace, delegate_to_method('pseudo_inverse')]

    @property
    def operand(self):
        return self.operands[0]

    def _expand(self):
        return self

    def pseudo_inverse(self):
        return self.operand

    def tex(self):
        return "{%s}%s" % (tex(self.operand), pseudo_dagger_tex)

    def __str__(self):
        if isinstance(self.operand, OperatorSymbol):
            return str(self.operand) + pseudo_dagger_str
        return "(%s)%s" % (self.operand, pseudo_dagger_str)


PseudoInverse._delegate_to_method += (PseudoInverse, )


class NullSpaceProjector(OperatorOperation):
    r"""Returns a projection operator :math:`\mathcal{P}_{{\rm Ker} X}` that
    projects onto the nullspace of its operand

    .. math::

        X \mathcal{P}_{{\rm Ker} X} = 0 \Leftrightarrow  X (1 - \mathcal{P}_{{\rm Ker} X}) = X\\
        \mathcal{P}_{{\rm Ker} X}^\dagger = \mathcal{P}_{{\rm Ker} X} = \mathcal{P}_{{\rm Ker} X}^2

    Use as:

        ``NullSpaceProjector(X)``

    :param X: Operator argument
    :type X: Operator
    """

    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def operand(self):
        return self.operands[0]

    def tex(self):
        return op_tex(r'\matchcal{P}', subscript=r'{\rm Ker}',
                      args=[tex(self.operand), ])

    def __str__(self):
        return op_str(r'P', subscript=r'Ker',
                      args=[tex(self.operand), ])


class OperatorTrace(Operator, Operation):
    r"""Take the (partial) trace of an operator :math:`X` over the degrees of
    freedom given by a Hilbert hs :math:`\mathcal{H}`:

    .. math::

        {\rm Tr}_{\mathcal{H}} X

    Use as:

        ``OperatorTrace(X, over_space=hs)``

    :param over_space: The degrees of freedom to trace over
    :type over_space: HilbertSpace
    :param op: The operator to take the trace of.
    :type op: Operator
    """
    signature = (HilbertSpace, Operator), {}
    _rules = []  # see end of module
    _simplifications = [implied_local_space(keys=['over_space', ]),
                        match_replace, ]

    def __init__(self, op, *, over_space):
        if isinstance(over_space, (int, str)):
            over_space = LocalSpace(over_space)
        assert isinstance(over_space, HilbertSpace)
        self._over_space = over_space
        super().__init__(op)

    @property
    def kwargs(self):
        return {'over_space': self._over_space}

    @property
    def operand(self):
        return self.operands[0]

    @property
    def space(self):
        s = self._over_space
        o = self.operand
        return o.space / s

    def _expand(self):
        s = self._over_space
        o = self.operand
        return OperatorTrace.create(o.expand(), over_space=s)

    def _series_expand(self, param, about, order):
        ope = self.operand.series_expand(param, about, order)
        return tuple(OperatorTrace.create(opet, over_space=self._over_space)
                     for opet in ope)

    def tex(self):
        s = self._over_space
        o = self.operand
        return r"{{\rm Tr}}_{{{}}} \left[ {} \right]".format(tex(s), tex(o))

    def __str__(self):
        s = self._over_space
        o = self.operand
        return r"tr_{!s}[{!s}]".format(s, o)

    def all_symbols(self):
        return self.operand.all_symbols()

    def _diff(self, sym):
        s = self._over_space
        o = self.operand
        return OperatorTrace.create(o._diff(sym), over_space=s)


tr = OperatorTrace.create


def factor_for_trace(ls, op):
    r"""Given a local space ls to take the partial trace over and an operator,
    factor the trace such that operators acting on disjoint degrees of freedom
    are pulled out of the trace. If the operator acts trivially on ls the trace
    yields only a pre-factor equal to the dimension of ls. If there are
    LocalSigma operators among a product, the trace's cyclical property is used
    to move to sandwich the full product by :py:class:`LocalSigma` operators:

    .. math::

        {\rm Tr} A \sigma_{jk} B = {\rm Tr} \sigma_{jk} B A \sigma_{jj}

    :param ls: Degree of Freedom to trace over
    :type ls: HilbertSpace
    :param op: Operator to take the trace of
    :type op: Operator
    :return: The (partial) trace over the operator's spc-degrees of freedom
    :rtype: Operator
    """
    if op.space == ls:
        if isinstance(op, OperatorTimes):
            pull_out = [o for o in op.operands if o.space is TrivialSpace]
            rest = [o for o in op.operands if o.space is not TrivialSpace]
            if pull_out:
                return (OperatorTimes.create(*pull_out) *
                        OperatorTrace.create(OperatorTimes.create(*rest),
                                             over_space=ls))
        raise CannotSimplify()
    if ls & op.space == TrivialSpace:
        return ls.get_dimension() * op
    if ls < op.space and isinstance(op, OperatorTimes):
        pull_out = [o for o in op.operands if (o.space & ls) == TrivialSpace]

        rest = [o for o in op.operands if (o.space & ls) != TrivialSpace]
        if (not isinstance(rest[0], LocalSigma) or
                not isinstance(rest[-1], LocalSigma)):
            found_ls = False
            for j, r in enumerate(rest):
                if isinstance(r, LocalSigma):
                    found_ls = True
                    break
            if found_ls:
                m, n = r.args[1:]
                rest = rest[j:] + rest[:j] + [LocalSigma(ls, m, m), ]
        if not rest:
            rest = [IdentityOperator]
        if len(pull_out):
            return (OperatorTimes.create(*pull_out) *
                    OperatorTrace.create(OperatorTimes.create(*rest),
                                         over_space=ls))
    raise CannotSimplify()


def decompose_space(H, A):
    """Simplifies OperatorTrace expressions over tensor-product spaces by
    turning it into iterated partial traces.

    :param H: The full space.
    :type H: ProductSpace
    :type A: Operator
    :return: Iterative partial trace expression
    :rtype: Operator
    """
    return OperatorTrace.create(
                OperatorTrace.create(A, over_space=H.operands[-1]),
                over_space=ProductSpace.create(*H.operands[:-1]))


## Expression rewriting _rules
u = wc("u", head=Operator.scalar_types)
v = wc("v", head=Operator.scalar_types)

n = wc("n", head=(int, str))
m = wc("m", head=(int, str))

A = wc("A", head=Operator)
A__ = wc("A__", head=Operator)
A___ = wc("A___", head=Operator)
B = wc("B", head=Operator)
B__ = wc("B__", head=Operator)
B___ = wc("B___", head=Operator)
C = wc("C", head=Operator)

A_plus = wc("A", head=OperatorPlus)
A_times = wc("A", head=OperatorTimes)
A_local = wc("A", head=LocalOperator)
B_local = wc("B", head=LocalOperator)

ls = wc("ls", head=LocalSpace)
h1 = wc("h1", head=HilbertSpace)
h2 = wc("h2", head=HilbertSpace)
H_ProductSpace = wc("H", head=ProductSpace)

ra = wc("ra", head=(int, str))
rb = wc("rb", head=(int, str))
rc = wc("rc", head=(int, str))
rd = wc("rd", head=(int, str))

ScalarTimesOperator._rules += [
    (pattern_head(1, A),
        lambda A: A),
    (pattern_head(0, A),
        lambda A: ZeroOperator),
    (pattern_head(u, ZeroOperator),
        lambda u: ZeroOperator),
    (pattern_head(u, pattern(ScalarTimesOperator, v, A)),
        lambda u, v, A: (u * v) * A)
]

OperatorPlus._binary_rules += [
    (pattern_head(pattern(ScalarTimesOperator, u, A),
                  pattern(ScalarTimesOperator, v, A)),
        lambda u, v, A: (u + v) * A),
    (pattern_head(pattern(ScalarTimesOperator, u, A), A),
        lambda u, A: (u + 1) * A),
    (pattern_head(A, pattern(ScalarTimesOperator, v, A)),
        lambda v, A: (1 + v) * A),
    (pattern_head(A, A),
        lambda A: 2 * A),
]


def Jpjmcoeff(ls, m):
    try:
        j = sympify(ls.get_dimension()-1)/2
        # m = j-m
        coeff = sqrt(j*(j+1)-m*(m+1))
        return coeff
    except BasisNotSetError:
        raise CannotSimplify()


def Jzjmcoeff(ls, m):
    try:
        return m
    except BasisNotSetError:
        raise CannotSimplify()


def Jmjmcoeff(ls, m):
    return Jpjmcoeff(ls, m-1)


OperatorTimes._binary_rules += [
    (pattern_head(pattern(ScalarTimesOperator, u, A), B),
        lambda u, A, B: u * (A * B)),

    (pattern_head(ZeroOperator, B),
        lambda B: ZeroOperator),
    (pattern_head(A, ZeroOperator),
        lambda A: ZeroOperator),

    (pattern_head(A, pattern(ScalarTimesOperator, u, B)),
        lambda A, u, B: u * (A * B)),

    (pattern_head(pattern(LocalSigma, ls, ra, rb),
                  pattern(LocalSigma, ls, rc, rd)),
        lambda ls, ra, rb, rc, rd: LocalSigma(ls, ra, rd)
     if rb == rc else ZeroOperator),


    # Harmonic oscillator rules
    (pattern_head(pattern(Create, ls), pattern(LocalSigma, ls, rc, rd)),
        lambda ls, rc, rd: sqrt(rc + 1) * LocalSigma(ls, rc + 1, rd)),

    (pattern_head(pattern(Destroy, ls), pattern(LocalSigma, ls, rc, rd)),
        lambda ls, rc, rd: sqrt(rc) * LocalSigma(ls, rc - 1, rd)),

    (pattern_head(pattern(LocalSigma, ls, rc, rd), pattern(Destroy, ls)),
        lambda ls, rc, rd: sqrt(rd + 1) * LocalSigma(ls, rc, rd + 1)),

    (pattern_head(pattern(LocalSigma, ls, rc, rd), pattern(Create, ls)),
        lambda ls, rc, rd: sqrt(rd) * LocalSigma(ls, rc, rd - 1)),

    # Normal ordering for harmonic oscillator <=> all a^* to the left, a to
    # the right.
    (pattern_head(pattern(Destroy, ls), pattern(Create, ls)),
        lambda ls: IdentityOperator + Create(ls) * Destroy(ls)),

    # Oscillator unitary group rules
    (pattern_head(pattern(Phase, ls, u), pattern(Phase, ls, v)),
        lambda ls, u, v: Phase.create(ls, u + v)),
    (pattern_head(pattern(Displace, ls, u), pattern(Displace, ls, v)),
        lambda ls, u, v: (exp((u * v.conjugate() - u.conjugate() * v) / 2) *
                          Displace.create(ls, u + v))),

    (pattern_head(pattern(Destroy, ls), pattern(Phase, ls, u)),
        lambda ls, u: exp(I * u) * Phase(ls, u) * Destroy(ls)),
    (pattern_head(pattern(Destroy, ls), pattern(Displace, ls, u)),
        lambda ls, u: Displace(ls, u) * (Destroy(ls) + u)),

    (pattern_head(pattern(Phase, ls, u), pattern(Create, ls)),
        lambda ls, u: exp(I * u) * Create(ls) * Phase(ls, u)),
    (pattern_head(pattern(Displace, ls, u), pattern(Create, ls)),
        lambda ls, u: (Create(ls) - u.conjugate()) * Displace(ls, u)),

    (pattern_head(pattern(Phase, ls, u), pattern(LocalSigma, ls, n, m)),
            lambda ls, u, n, m: exp(I * u * n) * LocalSigma(ls, n, m)),
    (pattern_head(pattern(LocalSigma, ls, n, m), pattern(Phase, ls, u)),
            lambda ls, u, n, m: exp(I * u * m) * LocalSigma(ls, n, m)),


    # Spin rules
     (pattern_head(pattern(Jplus, ls), pattern(LocalSigma, ls, rc, rd)),
        lambda ls, rc, rd: Jpjmcoeff(ls, rc) * LocalSigma(ls, rc + 1, rd)),

     (pattern_head(pattern(Jminus, ls), pattern(LocalSigma, ls, rc, rd)),
        lambda ls, rc, rd: Jmjmcoeff(ls, rc) * LocalSigma(ls, rc - 1, rd)),

    (pattern_head(pattern(Jz, ls), pattern(LocalSigma, ls, rc, rd)),
        lambda ls, rc, rd: rc * LocalSigma(ls, rc , rd)),

    (pattern_head(pattern(LocalSigma, ls, rc, rd), pattern(Jplus, ls)),
        lambda ls, rc, rd: Jmjmcoeff(ls, rd) * LocalSigma(ls, rc, rd - 1)),

    (pattern_head(pattern(LocalSigma, ls, rc, rd), pattern(Jminus, ls)),
        lambda ls, rc, rd: Jpjmcoeff(ls, rd) * LocalSigma(ls, rc, rd + 1)),

    (pattern_head(pattern(LocalSigma, ls, rc, rd), pattern(Jz, ls)),
        lambda ls, rc, rd: rd * LocalSigma(ls, rc, rd)),

    # Normal ordering for angular momentum <=> all J_+ to the left, J_z to
    # center and J_- to the right
    (pattern_head(pattern(Jminus, ls), pattern(Jplus, ls)),
        lambda ls: -2*Jz(ls) + Jplus(ls) * Jminus(ls)),

    (pattern_head(pattern(Jminus, ls), pattern(Jz, ls)),
        lambda ls: Jz(ls) * Jminus(ls) + Jminus(ls)),

    (pattern_head(pattern(Jz, ls), pattern(Jplus, ls)),
        lambda ls: Jplus(ls) * Jz(ls) + Jplus(ls)),


]

Adjoint._rules += [
    (pattern_head(pattern(ScalarTimesOperator, u, A)),
        lambda u, A: u.conjugate() * A.adjoint()),
    (pattern_head(A_plus),
        lambda A: OperatorPlus.create(*[o.adjoint()
                                        for o in A.operands])),
    (pattern_head(A_times),
        lambda A: OperatorTimes.create(*[o.adjoint()
                                         for o in A.operands[::-1]])),
    (pattern_head(pattern(Adjoint, A)),
        lambda A: A),
    (pattern_head(pattern(Create, ls)),
        lambda ls: Destroy(ls)),
    (pattern_head(pattern(Destroy, ls)),
        lambda ls: Create(ls)),
    (pattern_head(pattern(Jplus, ls)),
        lambda ls: Jminus(ls)),
    (pattern_head(pattern(Jminus, ls)),
        lambda ls: Jplus(ls)),
    (pattern_head(pattern(Jz, ls)),
        lambda ls: Jz(ls)),
    (pattern_head(pattern(LocalSigma, ls, ra, rb)),
        lambda ls, ra, rb: LocalSigma(ls, rb, ra)),
]

Displace._rules += [
    (pattern_head(ls, 0), lambda ls: IdentityOperator)
]
Phase._rules += [
    (pattern_head(ls, 0), lambda ls: IdentityOperator)
]
Squeeze._rules += [
    (pattern_head(ls, 0), lambda ls: IdentityOperator)
]

OperatorTrace._rules += [
    (pattern_head(A, over_space=TrivialSpace),
        lambda A: A),
    (pattern_head(ZeroOperator, over_space=h1),
        lambda h1: ZeroOperator),
    (pattern_head(IdentityOperator, over_space=h1),
        lambda h1: h1.get_dimension() * IdentityOperator),
    (pattern_head(A_plus, over_space=h1),
        lambda h1, A: OperatorPlus.create(
            *[OperatorTrace.create(o, over_space=h1) for o in A.operands])),
    (pattern_head(pattern(Adjoint, A), over_space=h1),
        lambda h1, A: Adjoint.create(OperatorTrace.create(A, over_space=h1))),
    (pattern_head(pattern(ScalarTimesOperator, u, A), over_space=h1),
        lambda h1, u, A: u * OperatorTrace.create(A, over_space=h1)),
    (pattern_head(A, over_space=H_ProductSpace),
        lambda H, A: decompose_space(H, A)),
    (pattern_head(pattern(Create, ls), over_space=ls),
        lambda ls: ZeroOperator),
    (pattern_head(pattern(Destroy, ls), over_space=ls),
        lambda ls: ZeroOperator),
    (pattern_head(pattern(LocalSigma, ls, n, m), over_space=ls),
        lambda ls, n, m: IdentityOperator if n == m else ZeroOperator),
    (pattern_head(A, over_space=ls),
        lambda ls, A: factor_for_trace(ls, A)),
]

PseudoInverse._rules += [
    (pattern_head(pattern(LocalSigma, ls, m, n)),
        lambda ls, m, n: LocalSigma(ls, n, m)),
]


class NonSquareMatrix(Exception):
    pass


class Matrix(Expression):
    """Matrix with Operator (or scalar-) valued elements."""
    matrix = None
    _hash = None

    def __init__(self, m):
        if isinstance(m, ndarray):
            self.matrix = m
        elif isinstance(m, Matrix):
            self.matrix = np_array(m.matrix)
        else:
            self.matrix = np_array(m)
        if len(self.matrix.shape) < 2:
            self.matrix = self.matrix.reshape((self.matrix.shape[0], 1))
        if len(self.matrix.shape) > 2:
            raise ValueError()

    @property
    def shape(self):
        """The shape of the matrix ``(nrows, ncols)``"""
        return self.matrix.shape

    @property
    def block_structure(self):
        """For square matrices this gives the block (-diagonal) structure of
        the matrix as a tuple of integers that sum up to the full dimension.

        :type: tuple
        """
        n, m = self.shape
        if n != m:
            raise AttributeError("block_structure only defined for square matrices")
        for k in range(1, n):
            if (self.matrix[:k, k:] == 0).all() and (self.matrix[k:, :k] == 0).all():
                return (k,) + self[k:, k:].block_structure
        return n,

    def _get_blocks(self, block_structure):
        n, m = self.shape
        if n == m:
            if not sum(block_structure) == n:
                raise ValueError()
            if not len(block_structure):
                return ()
            j = block_structure[0]

            if (self.matrix[:j, j:] == 0).all() and (self.matrix[j:, :j] == 0).all():
                return (self[:j, :j],) + self[j:, j:]._get_blocks(block_structure[1:])
            else:
                raise ValueError()
        elif m == 1:
            if not len(block_structure):
                return ()
            else:
                return (self[:block_structure[0], :],) + self[:block_structure[0], :]._get_blocks(block_structure[1:])
        else:
            raise ValueError()

    @property
    def args(self):
        return (self.matrix, )

    def __hash__(self):
        if not self._hash:
            self._hash = hash((tuple(self.matrix.flatten()),
                               self.matrix.shape, Matrix))
        return self._hash

    def __eq__(self, other):
        return (isinstance(other, Matrix)
                and self.shape == other.shape
                and (self.matrix == other.matrix).all())

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.matrix + other.matrix)
        else:
            return Matrix(self.matrix + other)

    def __radd__(self, other):
        return Matrix(other + self.matrix)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.matrix.dot(other.matrix))
        else:
            return Matrix(self.matrix * other)

    def __rmul__(self, other):
        return Matrix(other * self.matrix)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne / other)
        raise NotImplementedError("Can't divide matrix %s by %s"
                                  % (self, other))

    __truediv__ = __div__

    #    def __pow__(self, power):
    #        return OperatorMatrix(self.matrix.__pow__(power))

    def transpose(self):
        """The transpose matrix"""
        return Matrix(self.matrix.T)

    def conjugate(self):
        """The element-wise conjugate matrix, i.e., if an element is an
        operator this means the adjoint operator, but no transposition of
        matrix elements takes place.
        """
        return Matrix(np_conjugate(self.matrix))

    @property
    def T(self):
        """Transpose matrix"""
        return self.transpose()

    def adjoint(self):
        """Return the adjoint operator matrix, i.e. transpose and the Hermitian
        adjoint operators of all elements.
        """
        return self.T.conjugate()

    dag = adjoint

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.matrix.tolist())

    def __str__(self):
        return "{}({!s})".format(self.__class__.__name__, self.matrix)

    def trace(self):
        if self.shape[0] == self.shape[1]:
            return sum(self.matrix[k, k] for k in range(self.shape[0]))
        raise NonSquareMatrix(repr(self))

    @property
    def H(self):
        """Return the adjoint operator matrix, i.e. transpose and the Hermitian
        adjoint operators of all elements."""
        return self.adjoint()

    def __getitem__(self, item_id):
        item = self.matrix.__getitem__(item_id)
        if isinstance(item, ndarray):
            return Matrix(item)
        return item

    def element_wise(self, method):
        """Apply a method to each matrix element and return the result in a new
        operator matrix of the same shape.  :param method: A method taking a
        single argument.
        :type method: FunctionType
        :return: Operator matrix with results of method applied element-wise.
        :rtype: Matrix
        """
        s = self.shape
        emat = [method(o) for o in self.matrix.flatten()]
        return Matrix(np_array(emat).reshape(s))

    def series_expand(self, param, about, order):
        """Expand the matrix expression as a truncated power series in a scalar
        parameter.

        :param param: Expansion parameter.
        :type param: sympy.core.symbol.Symbol
        :param about: Point about which to expand.
        :type about:  Any one of Operator.scalar_types
        :param order: Maximum order of expansion.
        :type order: int >= 0
        :return: tuple of length (order+1), where the entries are the expansion coefficients.
        :rtype: tuple of Operator
        """
        s = self.shape
        emats = zip(*[o.series_expand(param, about, order)
                      for o in self.matrix.flatten()])
        return tuple((Matrix(np_array(em).reshape(s)) for em in emats))

    def expand(self):
        """Expand each matrix element distributively.
        :return: Expanded matrix.
        :rtype: Matrix
        """
        m = lambda o: o.expand() if isinstance(o, Operator) else o
        return self.element_wise(m)

    def _substitute(self, var_map):

        def _substitute(o):
            sympy_var_map = {k: v for (k, v) in var_map.items()
                             if isinstance(k, SympyBasic)}
            if isinstance(o, Operation):
                return substitute(o, var_map)
            elif isinstance(o, SympyBasic):
                return o.subs(sympy_var_map)
            else:
                return o

        return self.element_wise(_substitute)

    def all_symbols(self):
        ret = set()
        for o in self.matrix.flatten():
            if isinstance(o, Operator):
                ret = ret | o.all_symbols()
            else:
                ret = ret | scalar_free_symbols(o)
        return ret

    def tex(self):
        ret = r"\begin{pmatrix} "
        #        for row in self.matrix:
        ret += r""" \\ """.join([" & ".join([tex(o) for o in row])
                                 for row in self.matrix])
        ret += r"\end{pmatrix}"

        return ret

    @property
    def space(self):
        """Combined Hilbert space of all matrix elements."""
        arg_spaces = [o.space for o in self.matrix.flatten()
                      if hasattr(o, 'space')]
        if len(arg_spaces) == 0:
            return TrivialSpace
        else:
            return ProductSpace.create(*arg_spaces)

    def simplify_scalar(self):
        """
        Simplify all scalar expressions appearing in the Matrix.
        """
        return self.element_wise(simplify_scalar)


def hstackm(matrices):
    """Generalizes `numpy.hstack` to OperatorMatrix objects."""
    return Matrix(np_hstack(tuple(m.matrix for m in matrices)))


def vstackm(matrices):
    """Generalizes `numpy.vstack` to OperatorMatrix objects."""
    arr = np_vstack(tuple(m.matrix for m in matrices))
    #    print(tuple(m.matrix.dtype for m in matrices))
    #    print(arr.dtype)
    return Matrix(arr)


def diagm(v, k=0):
    """Generalizes the diagonal matrix creation capabilities of `numpy.diag` to
    OperatorMatrix objects."""
    return Matrix(np_diag(v, k))


def block_matrix(A, B, C, D):
    r"""Generate the operator matrix with quadrants

    .. math::

       \begin{pmatrix} A B \\ C D \end{pmatrix}

    :param A: Matrix of shape ``(n, m)``
    :type A: Matrix
    :param B: Matrix of shape ``(n, k)``
    :type B: Matrix
    :param C: Matrix of shape ``(l, m)``
    :type C: Matrix
    :param D: Matrix of shape ``(l, k)``
    :type D: Matrix

    :return: The combined block matrix [[A, B], [C, D]].
    :type: OperatorMatrix
    """
    return vstackm((hstackm((A, B)), hstackm((C, D))))


def identity_matrix(N):
    """Generate the N-dimensional identity matrix.

    :param N: Dimension
    :type N: int
    :return: Identity matrix in N dimensions
    :rtype: Matrix
    """
    return diagm(np_ones(N, dtype=int))


def zerosm(shape, *args, **kwargs):
    """Generalizes ``numpy.zeros`` to :py:class:`Matrix` objects."""
    return Matrix(np_zeros(shape, *args, **kwargs))


def permutation_matrix(permutation):
    r"""Return an orthogonal permutation matrix
    :math:`M_\sigma`
    for a permutation :math:`\sigma` defined by the image tuple
    :math:`(\sigma(1), \sigma(2),\dots \sigma(n))`,
    such that

    .. math::

        M_\sigma \vec{e}_i = \vec{e}_{\sigma(i)}

    where :math:`\vec{e}_k` is the k-th standard basis vector.
    This definition ensures a composition law:

    .. math::

        M_{\sigma \cdot \tau} = M_\sigma M_\tau.

    The column form of :math:`M_\sigma` is thus given by

    .. math::

        M = (\vec{e}_{\sigma(1)}, \vec{e}_{\sigma(2)}, \dots \vec{e}_{\sigma(n)}).

    :param permutation: A permutation image tuple (zero-based indices!)
    :type permutation: tuple
    """
    assert (check_permutation(permutation))
    n = len(permutation)
    op_matrix = np_zeros((n, n), dtype=int)
    for i, j in enumerate(permutation):
        op_matrix[j, i] = 1
    return Matrix(op_matrix)

# :deprecated:
# for backwards compatibility
OperatorMatrixInstance = Matrix
IdentityMatrix = identity_matrix


def Im(op):
    """The imaginary part of a number or operator. Acting on OperatorMatrices,
    it produces the element-wise imaginary parts.

    :param op: Anything that has a conjugate method.
    :type op: Operator or Matrix or any of Operator.scalar_types
    :return: The imaginary part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() - op) * I / 2


def Re(op):
    """The real part of a number or operator. Acting on OperatorMatrices, it
    produces the element-wise real parts.

    :param op: Anything that has a conjugate method.
    :type op: Operator or Matrix or any of Operator.scalar_types
    :return: The real part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() + op) / 2


def ImAdjoint(opmatrix):
    """
    The imaginary part of an OperatorMatrix, i.e. a Hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: Matrix
    :return: The matrix imaginary part of the operand.
    :rtype: Matrix
    """
    return (opmatrix.H - opmatrix) * I / 2


def ReAdjoint(opmatrix):
    """The real part of an OperatorMatrix, i.e. a Hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: Matrix
    :return: The matrix real part of the operand.
    :rtype: Matrix
    """
    return (opmatrix.H + opmatrix) / 2


def get_coeffs(expr, expand=False, epsilon=0.):
    """Create a dictionary with all Operator terms of the expression
    (understood as a sum) as keys and their coefficients as values.

    The returned object is a defaultdict that return 0. if a term/key
    doesn't exist.
    :param expr: The operator expression to get all coefficients from.
    :param expand: Whether to expand the expression distributively.
    :param epsilon: If non-zero, drop all Operators with coefficients that have absolute value less than epsilon.
    :return: A dictionary of {op1: coeff1, op2: coeff2, ...}
    :rtype: dict
    """
    if expand:
        expr = expr.expand()
    ret = defaultdict(int)
    operands = expr.operands if isinstance(expr, OperatorPlus) else [expr]
    for e in operands:
        c, t = _coeff_term(e)
        try:
            if abs(complex(c)) < epsilon:
                continue
        except TypeError:
            pass
        ret[t] += c
    return ret
