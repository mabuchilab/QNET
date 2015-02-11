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

This module features classes and functions to define and manipulate symbolic Operator expressions.
For more details see :ref:`operator_algebra`.

For a list of all properties and methods of an operator object, see the documentation for the basic :py:class:`Operator` class.
"""
# TODO extend slightly, add code examples


from __future__ import division
from abc import ABCMeta, abstractproperty, abstractmethod
from collections import defaultdict
from itertools import product as cartesian_product

import qutip

import qnet.algebra.abstract_algebra
from qnet.algebra.hilbert_space_algebra import *


#noinspection PyUnresolvedReferences
from sympy import (exp, log, cos, sin, cosh, sinh, tan, cot,
                   acos, asin, acosh, asinh, atan, atan2, atanh, acot, sqrt,
                   factorial, pi, I, sympify,
                   Matrix as SympyMatrix,
                   Basic as SympyBasic, symbols, Mul, Add, series as sympy_series)
from sympy.printing import latex as sympy_latex

#noinspection PyUnresolvedReferences
from numpy import (array as np_array,
                   shape as np_shape,
                   hstack as np_hstack,
                   vstack as np_vstack,
                   diag as np_diag,
                   ones as np_ones,
                   conjugate as np_conjugate,
                   zeros as np_zeros,
                   ndarray,
                   arange,
                   cos as np_cos,
                   sin as np_sin,
                   eye as np_eye, 
                   argwhere,
                   int64, complex128, float64)
from qnet.algebra.permutations import check_permutation

sympyOne = sympify(1)


def adjoint(obj):
    """
    Return the adjoint of an obj.
    """
    try:
        return obj.adjoint()
    except AttributeError:
        return obj.conjugate()

@six.add_metaclass(ABCMeta)
class Operator(object):
    """
    The basic operator class, which fixes the abstract interface of operator objects 
    and where possible also defines the default behavior under operations.
    Any operator contains an associated HilbertSpace object, 
    on which it is taken to act non-trivially.
    """

    # which data types may serve as scalar coefficients
    scalar_types = int, long, float, complex, SympyBasic, int64, complex128, float64


    @property
    def space(self):
        """
        The Hilbert space associated with the operator on which it acts non-trivially

        :type: HilbertSpace
        """
        return self._space

    @abstractproperty
    def _space(self):
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """
        :return: The Hermitian adjoint of the operator.
        :rtype: Operator
        """
        return self._adjoint()

    def _adjoint(self):
        return Adjoint.create(self)

    conjugate = dag = adjoint

    def pseudo_inverse(self):
        """
        :return: The pseudo-Inverse of the Operator, i.e., it inverts the operator on the orthogonal complement of its nullspace
        :rtype: Operator
        """
        return self._pseudo_inverse()

    def _pseudo_inverse(self):
        return PseudoInverse.create(self)

    def to_qutip(self, full_space=None):
        """
        Create a numerical representation of the operator as a QuTiP object.
        Note that all symbolic scalar parameters need to be replaced by numerical values before calling this method.

        :param full_space: The full Hilbert space in which to represent the operator.
        :type full_space: HilbertSpace
        :return: The matrix representation of the operator.
        :rtype: qutip.Qobj (:py:class:`qutip.Qobj`)
        """
        if full_space is None:
            return self._to_qutip(self.space)
        else:
            return self._to_qutip(full_space)

    @abstractmethod
    def _to_qutip(self, full_space):
        raise NotImplementedError(str(self.__class__))

    def expand(self):
        """
        Expand out distributively all products of sums. Note that this does not expand out sums of scalar coefficients.

        :return: A fully expanded sum of operators.
        :rtype: Operator
        """
        return self._expand()

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)


    def simplify_scalar(self):
        """
        Simplify all scalar coefficients within the Operator expression.

        :return: The simplified expression.
        :rtype: Operator
        """
        return self._simplify_scalar()

    def _simplify_scalar(self):
        return self

    def series_expand(self, param, about, order):
        """
        Expand the operator expression as a truncated power series in a scalar parameter.

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
            return prod((self,) * other, 1)
        else:
            return NotImplemented


def space(obj):
    """
    Gives the associated HilbertSpace with an object. Also works for :py:attr:`Operator.scalar_types`.

    :type obj: Operator or Operator.scalar_types
    :rtype: HilbertSpace
    """
    try:
        return obj.space
    except AttributeError:
        if isinstance(obj, Operator.scalar_types):
            return TrivialSpace
        raise ValueError(str(obj))


@check_signature
class OperatorSymbol(Operator, Operation):
    """
    Operator Symbol class, parametrized by an identifier string and an associated Hilbert space.

        ``OperatorSymbol(name, hs)``

    :param name: Symbol identifier
    :type name: basestring
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace
    """
    signature = basestring, (HilbertSpace, basestring, int, tuple)

    def __init__(self, name, hs):
        if isinstance(hs, (str, int)):
            #noinspection PyArgumentList
            super(OperatorSymbol, self).__init__(name, local_space(hs))
        elif isinstance(hs, tuple):
            #noinspection PyArgumentList
            super(OperatorSymbol, self).__init__(name, prod([local_space(h) for h in hs], neutral=TrivialSpace))
        else:
            #noinspection PyArgumentList
            super(OperatorSymbol, self).__init__(name, hs)


    def __str__(self):
        return self.operands[0]

    def _tex(self):
        return identifier_to_tex(self.operands[0])

    def _to_qutip(self, full_space=None):
        raise AlgebraError("Cannot convert operator symbol to representation matrix. Substitute first.")

    @property
    def _space(self):
        return self.operands[1]

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def _all_symbols(self):
        return {self}


@singleton
class IdentityOperator(Operator, Expression):
    """
    ``IdentityOperator`` constant (singleton) object.
    """

    @property
    def _space(self):
        return TrivialSpace

    def _adjoint(self):
        return self

    def _to_qutip(self, full_space):
        return qutip.tensor(*[qutip.qeye(s.dimension) for s in full_space.local_factors()])

    #    def mathematica(self):
    #        return "IdentityOperator"

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def _pseudo_inverse(self):
        return self

    def _tex(self):
        return "1"

    def __str__(self):
        return "II"

    def __eq__(self, other):
        return self is other  # or other == 1

    def _all_symbols(self):
        return set(())


II = IdentityOperator

from scipy.sparse import csr_matrix


@singleton
class ZeroOperator(Operator, Expression):
    """
    ``ZeroOperator`` constant (singleton) object.
    """

    @property
    def _space(self):
        return TrivialSpace

    def _adjoint(self):
        return self

    def _to_qutip(self, full_space):
        return qutip.tensor(
            *[qutip.Qobj(csr_matrix((s.dimension, s.dimension))) for s in full_space.local_factors()])

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def _pseudo_inverse(self):
        return self

    def _tex(self):
        return "0"

    def __eq__(self, other):
        return self is other or other == 0

    def __str__(self):
        return "0"

    def _all_symbols(self):
        return (())


def _implied_local_space_mtd(dcls, clsmtd, cls, space, *ops):
    """
    For Operations whose first operand is a local space, accept ``int`` or ``str`` arguments and convert these to a ``LocalSpace`` via :py:func:`qnet.algebra.hilbert_space_algebra.local_space`.
    """
    if isinstance(space, (int, str)):
        return clsmtd(cls, local_space(space), *ops)
    return clsmtd(cls, space, *ops)


implied_local_space = preprocess_create_with(_implied_local_space_mtd)


class LocalOperator(Operator, Operation):
    """
    Base class for all kinds of operators that act *locally*,
    i.e. only on a single degree of freedom.
    """

    def __init__(self, hs, *args):
        if isinstance(hs, (str, int)):
            hs = local_space(hs)
        super(LocalOperator, self).__init__(hs, *args)

    @property
    def _space(self):
        return self.operands[0]

    def _to_qutip(self, full_space=None):
        if full_space is None or full_space == self.space:
            return self.to_qutip_local_factor()
        else:
            all_spaces = full_space.local_factors()
            own_space_index = all_spaces.index(self.space)
            return qutip.tensor(*([qutip.qeye(s.dimension) for s in all_spaces[:own_space_index]]
                                  + [self.to_qutip_local_factor()]
                                  + [qutip.qeye(s.dimension) for s in all_spaces[own_space_index + 1:]]))

    def to_qutip_local_factor(self):
        """
        :return: Return a qutip representation for the local operator only on its local space.
        :rtype: qutip.Qobj
        """
        return self._to_qutip_local_factor()

    def _to_qutip_local_factor(self):
        raise NotImplementedError(self.__class__.__name__)

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def _all_symbols(self):
        return set()


@implied_local_space
@check_signature
class Create(LocalOperator):
    """
    ``Create(space)`` yields a bosonic creation operator acting on a particular local space/degree of freedom.
    Its adjoint is

        >>> Create(1).adjoint()
            Destroy(1)

    and it obeys the bosonic commutation relation

        >>> Destroy(1) * Create(1) - Create(1) * Destroy(1)
            1
        >>> Destroy(1) * Create(2) - Create(2) * Destroy(1)
            1

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    """
    signature = (LocalSpace, basestring, int),

    def _to_qutip_local_factor(self):
        return qutip.create(self.space.dimension)

    def _tex(self):
        return r"{{a_{{{}}}^\dagger}}".format(self.space.tex())

    def __str__(self):
        return r"(a_{!s})^*".format(self.space)


@implied_local_space
@check_signature
class Destroy(LocalOperator):
    """
    ``Destroy(space)`` yields a bosonic annihilation operator acting on a particular local space/degree of freedom.
    Its adjoint is

        >>> Destroy(1).adjoint()
            Create(1)

    and it obeys the bosonic commutation relation

        >>> Destroy(1) * Create(1) - Create(1) * Destroy(1)
            1
        >>> Destroy(1) * Create(2) - Create(2) * Destroy(1)
            1

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or str
    """

    signature = (LocalSpace, basestring, int),

    def _to_qutip_local_factor(self):
        return qutip.destroy(self.space.dimension)

    def _tex(self):
        return r"{{a_{{{}}}}}".format(self.space.tex())

    def __str__(self):
        return r"a_{!s}".format(self.space)


def simplify_scalar(s):
    """
    Simplify all occurences of scalar expressions in s

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
    if len(operands) > 1:
        return set_union([scalar_free_symbols(o) for o in operands])
    if len(operands) < 1:
        return set()
    o, = operands
    if isinstance(o, SympyBasic):
        return o.free_symbols
    return set()


@implied_local_space
@match_replace
@check_signature
class Phase(LocalOperator):
    r"""
    The unitary Phase operator acting on a particular local space/degree of freedom:

    .. math::

        P_{\rm s}(\phi):= \exp\left(i \phi a_{\rm s}^\dagger a_{\rm s}\right)

    where :math:`a_{\rm s}` is the annihilation operator acting on the local space s.
    Use as:

        ``Phase(space, phi)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or basestring
    :param phi: Displacement amplitude.
    :type phi: Any from `Operator.scalar_types`
    """
    signature = (LocalSpace, basestring, int), Operator.scalar_types
    _rules = []

    # TODO implement _series_expand for the phase parameter

    def _to_qutip_local_factor(self):
        arg = complex(self.operands[1]) * arange(self.space.dimension)
        d = np_cos(arg) + 1j * np_sin(arg)
        return qutip.Qobj(np_diag(d))

    def _adjoint(self):
        return Phase(self.operands[0], -self.operands[1].conjugate())

    def _pseudo_inverse(self):
        return Phase(self.operands[0], -self.operands[1])

    def _tex(self):
        return r"{{P_{{{}}}({})}}".format(self.space.tex(), tex(self.operands[1]))

    def __str__(self):
        return "P_{!s}({!s})".format(self.space, self.operands[1])


    def _simplify_scalar(self):
        return Phase(self.space, simplify_scalar(self.operands[1]))

    def _all_symbols(self):
        return scalar_free_symbols(self.operands[0])


@implied_local_space
@match_replace
@check_signature
class Displace(LocalOperator):
    r"""
    Unitary coherent displacement operator

    .. math::

        D_{\rm s}(\alpha) := \exp\left({\alpha a_{\rm s}^\dagger - \alpha^* a_{\rm s}}\right)

    where :math:`a_{\rm s}` is the annihilation operator acting on the local space s.
    Use as:

        ``Displace(space, alpha)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or basestring
    :param alpha: Displacement amplitude.
    :type alpha: Any from `Operator.scalar_types`
    """
    signature = (LocalSpace, basestring, int), Operator.scalar_types
    _rules = []

    # TODO implement _series_expand for the coherent displacement parameter

    def _to_qutip_local_factor(self):
        return qutip.displace(self.space.dimension, complex(self.operands[1]))

    def _adjoint(self):
        return Displace(self.operands[0], -self.operands[1])

    _pseudo_inverse = _adjoint

    def _tex(self):
        return r"{{D_{{{}}}({})}}".format(self.space.tex(), tex(self.operands[1]))

    def __str__(self):
        return "D_{!s}({!s})".format(self.space, self.operands[1])

    def _simplify_scalar(self):
        return Displace(self.space, simplify_scalar(self.operands[1]))

    def _all_symbols(self):
        return scalar_free_symbols(self.operands[0])


@implied_local_space
@match_replace
@check_signature
class Squeeze(LocalOperator):
    r"""
    A unitary Squeezing operator acting on a particular local space/degree of freedom:

    .. math::

        S_{\rm s}(\eta) := \exp {\left( \frac{\eta}{2} {a_{\rm s}^\dagger}^2 - \frac{\eta^*}{2} {a_{\rm s}}^2 \right)}

    where :math:`a_{\rm s}` is the annihilation operator acting on the local space s.
    Use as:

        ``Squeeze(space, eta)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or basestring
    :param eta: Squeeze parameter.
    :type eta: Any from `Operator.scalar_types`
    """
    signature = (LocalSpace, basestring, int), Operator.scalar_types
    _rules = []

    # TODO implement _series_expand for the squeeze parameter

    def _to_qutip_local_factor(self):
        return qutip.squeeze(self.space.dimension, complex(self.operands[1]))

    def _adjoint(self):
        return Squeeze(self.operands[0], -self.operands[1])

    _pseudo_inverse = _adjoint

    def _tex(self):
        return r"{{S_{{{}}}({})}}".format(self.space.tex(), tex(self.operands[1]))

    def __str__(self):
        return "S_{!s}({!s})".format(self.space, self.operands[1])

    def _simplify_scalar(self):
        return Squeeze(self.space, simplify_scalar(self.operands[1]))

    def _all_symbols(self):
        return scalar_free_symbols(self.operands[0])


@implied_local_space
@check_signature
class LocalSigma(LocalOperator):
    r"""
    A local level flip operator operator acting on a particular local space/degree of freedom.

    .. math::

        \sigma_{jk}^{\rm s} := \left| j\right\rangle_{\rm s} \left \langle k \right |_{\rm s}

    Use as:

        ``LocalSigma(space, j, k)``

    :param space: Associated local Hilbert space.
    :type space: LocalSpace or basestring
    :param j: State label j.
    :type j: int or basestring
    :param k: State label k.
    :type k: int or basestring
    """

    signature = (LocalSpace, basestring, int), (int, basestring), (int, basestring)

    def _to_qutip_local_factor(self):
        k, j = self.operands[1:]
        ket = qutip.basis(self.space.dimension, self.space.basis.index(k))
        bra = qutip.basis(self.space.dimension, self.space.basis.index(j)).dag()
        return ket * bra

    def _tex(self):
        j, k = self.operands[1:]
        if k == j:
            return r"{{\Pi_{{{}}}^{{{}}}}}".format(tex(k), self.space.tex())
        return r"{{\sigma_{{{},{}}}^{{{}}}}}".format(tex(j), tex(k), self.space.tex())

    def __str__(self):
        j, k = self.operands[1:]
        if k == j:
            return "Pi_{!s}^[{!s}]".format(j, self.space)
        return "sigma_{!s}{!s}^[{!s}]".format(j, k, self.space)


LocalProjector = lambda spc, state: LocalSigma.create(spc, state, state)


def X(local_space, states=("h", "g")):
    r"""
    Pauli-type X-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or basestring
    :return: Local X-operator.
    :rtype: Operator
    """
    h, g = states
    return LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h)


def Y(local_space, states=("h", "g")):
    r"""
    Pauli-type Y-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or basestring
    :return: Local Y-operator.
    :rtype: Operator
    """
    h, g = states
    return I * (-LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h))


def Z(local_space, states=("h", "g")):
    r"""
    Pauli-type Z-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or basestring
    :return: Local Z-operator.
    :rtype: Operator
    """
    h, g = states
    return LocalProjector(local_space, h) - LocalProjector(local_space, g)


class OperatorOperation(Operator, Operation):
    """
    Base class for Operations acting only on Operator arguments.
    """
    signature = Operator,

    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)

    def _simplify_scalar(self):
        return self.__class__.create(*[o.simplify_scalar() for o in self.operands])


def tuple_sum(tuples, inital=0):
    return tuple(sum(tels, 0) for tels in zip(*tuples))



@assoc
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_assoc
class OperatorPlus(OperatorOperation):
    """
    A sum of Operators

        ``OperatorPlus(*summands)``

    :param summands: Operator summands.
    :type summands: Operator
    """
    neutral_element = ZeroOperator
    _binary_rules = []

    @classmethod
    def order_key(cls, a):

        if isinstance(a, ScalarTimesOperator):
            c = a.coeff
            if isinstance(c, SympyBasic):
                c = inf
            return KeyTuple((Operation.order_key(a.term), c))
        return KeyTuple((Operation.order_key(a), 1))

    def _to_qutip(self, full_space=None):
        if full_space is None:
            full_space = self.space
        assert self.space <= full_space
        return sum((op.to_qutip(full_space) for op in self.operands), 0)

    def _expand(self):
        return sum((o.expand() for o in self.operands), ZeroOperator)

    def _series_expand(self, param, about, order):
        res = tuple_sum((o.series_expand(param, about, order) for o in self.operands))
        return res

    def _tex(self):
        ret = self.operands[0].tex()

        for o in self.operands[1:]:
            if isinstance(o, ScalarTimesOperator) and ScalarTimesOperator.has_minus_prefactor(o.coeff):
                ret += " - " + tex(-o)
            else:
                ret += " + " + tex(o)
        return ret

    def __str__(self):
        ret = str(self.operands[0])

        for o in self.operands[1:]:
            if isinstance(o, ScalarTimesOperator) and ScalarTimesOperator.has_minus_prefactor(o.coeff):
                ret += " - " + str(-o)
            else:
                ret += " + " + str(o)
        return ret


@assoc
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_assoc
class OperatorTimes(OperatorOperation):
    """
    A product of Operators that serves both as a product within a Hilbert space as well as a tensor product.

        ``OperatorTimes(*factors)``

    :param factors: Operator factors.
    :type factors: Operator
    """

    neutral_element = IdentityOperator
    _binary_rules = []

    class OperatorOrderKey(object):
        """
        Auxiliary class that generates the correct pseudo-order relation for operator products.
        Only operators acting on different Hilbert spaces are commuted to achieve the order specified in the full HilbertSpace.
        I.e., sorted(factors, key = OperatorOrderKey) achieves this ordering.
        """

        def __init__(self, op):
            s = op.space
            self.op = op
            self.full = False
            self.trivial = False
            self.local_spaces = set()
            if isinstance(s, LocalSpace):
                self.local_spaces = {s.operands, }
            elif s is TrivialSpace:
                self.trivial = True
            elif s is FullSpace:
                self.full = True
            else:
                # print(op, s)
                assert isinstance(s, ProductSpace)
                self.local_spaces = {s.operands for s in s.operands}

        def __lt__(self, other):
            if self.trivial and other.trivial:
                return Operation.order_key(self.op) < Operation.order_key(other.op)
            if self.full or len(self.local_spaces & other.local_spaces):
                return False
            return tuple(self.local_spaces) < tuple(other.local_spaces)

        def __gt__(self, other):
            if self.trivial and other.trivial:
                return Operation.order_key(self.op) > Operation.order_key(other.op)

            if self.full or len(self.local_spaces & other.local_spaces):
                return False

            return tuple(self.local_spaces) > tuple(other.local_spaces)

        def __eq__(self, other):
            if self.trivial and other.trivial:
                return Operation.order_key(self.op) == Operation.order_key(other.op)

            return self.full or len(self.local_spaces & other.local_spaces) > 0

    order_key = OperatorOrderKey

    @classmethod
    def create(cls, *ops):
        if any(o == ZeroOperator for o in ops):
            return ZeroOperator
        return cls(*ops)

    def factor_for_space(self, spc):
        if spc == TrivialSpace:
            ops_on_spc = [o for o in self.operands if o.space is TrivialSpace]
            ops_not_on_spc = [o for o in self.operands if o.space > TrivialSpace]
        else:
            ops_on_spc = [o for o in self.operands if (o.space & spc) > TrivialSpace]
            ops_not_on_spc = [o for o in self.operands if (o.space & spc) is TrivialSpace]
        return OperatorTimes.create(*ops_on_spc), OperatorTimes.create(*ops_not_on_spc)

    def _to_qutip(self, full_space=None):
        # if any factor acts non-locally, we need to expand distributively.
        if any(len(op.space) > 1 for op in self.operands):
            se = self.expand()
            if se == self:
                raise ValueError("Cannot represent as QuTiP object: {!s}".format(self))
            return se.to_qutip(full_space)

        if full_space == None:
            full_space = self.space

        all_spaces = full_space.local_factors()
        by_space = []
        ck = 0
        for ls in all_spaces:
            # group factors by associated local space
            ls_ops = [o.to_qutip() for o in self.operands if o.space == ls]
            if len(ls_ops):
                # compute factor associated with local space
                by_space.append(prod(ls_ops))
                ck += len(ls_ops)
            else:
                # if trivial action, take identity matrix
                by_space.append(qutip.qeye(ls.dimension))
        assert ck == len(self.operands)
        # combine local factors in tensor product
        return qutip.tensor(*by_space)

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, OperatorPlus) else (eo,) for eo in eops]
        # iterate over a cartesian product of all factor summands, form product of each tuple and sum over result
        ret = sum((OperatorTimes.create(*combo) for combo in cartesian_product(*eopssummands)), ZeroOperator)
        if isinstance(ret, OperatorPlus):
            return ret.expand()
        return ret

    def _series_expand(self, param, about, order):
        assert len(self.operands) > 1
        cfirst = self.operands[0].series_expand(param, about, order)
        crest = OperatorTimes.create(*self.operands[1:]).series_expand(param, about, order)
        return tuple(sum(cfirst[k] * crest[n - k] for k in range(n + 1)) for n in range(order + 1))

    def _tex(self):
        ret = ""
        # for o in self.operands[1:]:
        for o in self.operands:
            if isinstance(o, OperatorPlus):
                ret += r" \left({}\right) ".format(tex(o))
            else:
                ret += " {}".format(tex(o))
        return ret

    def __str__(self):
        # ret = str(self.operands[0])
        # for o in self.operands[1:]:
        #     if isinstance(o, OperatorPlus):
        #         ret += r" ({})".format(str(o))
        #     else:
        #         ret += " {}".format(str(o))
        ret = ""
        for o in self.operands:
            if isinstance(o, OperatorPlus):
                ret += r" ({}) ".format(str(o))
            else:
                ret += " {}".format(str(o))

        return ret


@match_replace
@check_signature
class ScalarTimesOperator(Operator, Operation):
    """
    Multiply an operator by a scalar coefficient.

        ``ScalarTimesOperator(coefficient, term)``

    :param coefficient: Scalar coefficient.
    :type coefficient: Any of Operator.scalar_types
    :param term: The operator that is multiplied.
    :type term: Operator
    """
    signature = Operator.scalar_types, Operator
    _rules = []

    @staticmethod
    def has_minus_prefactor(c):
        """
        For a scalar object c, determine whether it is prepended by a "-" sign.
        """
        cs = str(c).strip()
        return cs[0] == "-"


    @property
    def _space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]


    def _tex(self):
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

        return cs + ct

    def __str__(self):
        coeff, term = self.operands

        if isinstance(coeff, Add):
            cs = r"({!s})".format(coeff)
        else:
            cs = " {!s}".format(coeff)

        if ScalarTimesOperator.has_minus_prefactor(coeff):
            return " -" + str(-self)

        if term == IdentityOperator:
            ct = ""
        if isinstance(term, OperatorPlus):
            ct = r" ({!s})".format(term)
        else:
            ct = r" {!s}".format(term)

        return cs + ct


    def _to_qutip(self, full_space=None):
        return complex(self.coeff) * self.term.to_qutip(full_space)

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, OperatorPlus):
            return sum((c * eto for eto in et.operands), ZeroOperator)
        return c * et

    def _series_expand(self, param, about, order):
        te = self.term.series_expand(param, about, order)
        if isinstance(self.coeff, SympyBasic):
            if about != 0:
                c = self.coeff.subs({param: about + param})
            else:
                c = self.coeff
            try:
                ce = list(reversed(sympy_series(c, x=param, x0=0, n=order + 1).as_poly(param).all_coeffs()))
            except AttributeError:
                ce = [c] + [0] * order

            if len(ce) < order + 1:
                ce += [0] * (order + 1 - len(ce))
            return tuple(sum(ce[k] * te[n - k] for k in range(n + 1)) for n in range(order + 1))
        else:
            return tuple(self.coeff * tek for tek in te)

    def _pseudo_inverse(self):
        c, t = self.operands
        return t.pseudo_inverse() / c


    # def __complex__(self):
    #     if self.term is IdentityOperator:
    #         return complex(self.coeff)
    #     return NotImplemented

    # def __float__(self):
    #     if self.term is IdentityOperator:
    #         return float(self.coeff)
    #     return NotImplemented

    def __eq__(self, other):
        if self.term is IdentityOperator and isinstance(other, Operator.scalar_types):
            return self.coeff == other
        return super(ScalarTimesOperator, self).__eq__(other)


    def _substitute(self, var_map):
        st = self.term.substitute(var_map)
        if isinstance(self.coeff, SympyBasic):
            svar_map = {k: v for k, v in var_map.items() if not isinstance(k, Expression)}
            sc = self.coeff.subs(svar_map)
        else:
            sc = substitute(self.coeff, var_map)
        return sc * st


    def _simplify_scalar(self):
        coeff, term = self.operands
        return simplify_scalar(coeff) * term.simplify_scalar()

    def _all_symbols(self):
        return scalar_free_symbols(self.coeff) | self.term.all_symbols()


def safe_tex(obj):
    if isinstance(obj, (int, float, complex)):
        return format_number_for_tex(obj)

    if isinstance(obj, SympyBasic):
        return sympy_latex(obj).strip('$')
    try:
        return obj.tex()
    except AttributeError:
        return r"{\rm " + str(obj) + "}"


tex = qnet.algebra.abstract_algebra.tex = safe_tex


def format_number_for_tex(num):
    if num == 0:  #also True for 0., 0j
        return "0"
    if isinstance(num, complex):
        if num.real == 0:
            if num.imag == 1:
                return "i"
            if num.imag == -1:
                return "(-i)"
            if num.imag < 0:
                return "(-%si)" % format_number_for_tex(-num.imag)
            return "%si" % format_number_for_tex(num.imag)
        if num.imag == 0:
            return format_number_for_tex(num.real)
        return "(%s + %si)" % (format_number_for_tex(num.real), format_number_for_tex(num.imag))
    if num < 0:
        return "(%g)" % num
    return "%g" % num

#
#def format_number_for_mathematica(num):
#    if num == 0: #also True for 0., 0j
#        return "0"
#    if isinstance(num, complex):
#        if num.imag == 0:
#            return format_number_for_tex(num.real)
#        return "Complex[%g,%g]" % (num.real, num.imag)
#
#    return "%g" % num
#



greek_letter_strings = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
                        "zeta", "eta", "theta", "vartheta", "iota", "kappa",
                        "lambda", "mu", "nu", "xi", "pi", "varpi", "rho",
                        "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
                        "varphi", "chi", "psi", "omega",
                        "Gamma", "Delta", "Theta", "Lambda", "Xi",
                        "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]
greekToLatex = {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "delta": "Delta", "epsilon": "Epsilon",
                "varepsilon": "Epsilon",
                "zeta": "Zeta", "eta": "Eta", "theta": "Theta", "vartheta": "Theta", "iota": "Iota", "kappa": "Kappa",
                "lambda": "Lambda", "mu": "Mu", "nu": "Nu", "xi": "Xi", "pi": "Pi", "varpi": "Pi", "rho": "Rho",
                "varrho": "Rho", "sigma": "Sigma", "varsigma": "Sigma", "tau": "Tau", "upsilon": "Upsilon",
                "phi": "Phi",
                "varphi": "Phi", "chi": "Chi", "psi": "Psi", "omega": "Omega",
                "Gamma": "CapitalGamma", "Delta": "CapitalDelta", "Theta": "CapitalTheta", "Lambda": "CapitalLambda",
                "Xi": "CapitalXi",
                "Pi": "CapitalPi", "Sigma": "CapitalSigma", "Upsilon": "CapitalUpsilon", "Phi": "CapitalPhi",
                "Psi": "CapitalPsi", "Omega": "CapitalOmega"
}

import re

_idtp = re.compile(r'(?!\\)({})(\b|_)'.format("|".join(greek_letter_strings)))


def identifier_to_tex(identifier):
    """
    If an identifier contains a greek symbol name as a separate word, (e.g. ``my_alpha_1`` contains ``alpha`` as a separate word, but ``alphaman`` doesn't) add a backslash in front.

    :param identifier: The string to prepare for LaTeX printing
    :type identifier: str
    :returns: An improved version where greek letter symbols can be correctly rendered.
    :rtype: str
    """
    #    identifier = creduce(lambda a,b: "{%s_%s}" % (b, a), ["{%s}" % part for part in reversed(identifier.split("_"))])

    ret = _idtp.sub(r'{\\\1}\2', identifier)
    return ret


@check_signature
@match_replace
class Adjoint(OperatorOperation):
    """
    The symbolic Adjoint of an operator.

        ``Adjoint(op)``

    :param op: The operator to take the adjoint of.
    :type op: Operator
    """

    @property
    def operand(self):
        """
        :rtype: Operator
        """
        return self.operands[0]

    _rules = []

    def _to_qutip(self, full_space=None):
        return qutip.dag(self.operands[0].to_qutip(full_space))

    def _series_expand(self, param, about, order):
        ope = self.operand.series_expand(param, about, order)
        return tuple(adjoint(opet) for opet in ope)

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, OperatorPlus):
            return sum((adjoint(eoo) for eoo in eo.operands), ZeroOperator)
        return eo._adjoint()

    def _pseudo_inverse(self):
        return self.operand.pseudo_inverse().adjoint()

    def _tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^\dagger"

    def __str__(self):
        if isinstance(self.operand, OperatorSymbol):
            return "{}^*".format(str(self.operand))
        return "({})^*".format(str(self.operand))

# for hilbert space dimensions less than or equal to this,
# compute numerically PseudoInverse and NullSpaceProjector representations
DENSE_DIMENSION_LIMIT = 1000


@check_signature
@match_replace
class PseudoInverse(OperatorOperation):
    r"""
    The symbolic pseudo-inverse :math:`X^+` of an operator :math:`X`. It is defined via the relationship

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
    delegate_to_method = ScalarTimesOperator, Squeeze, Displace, ZeroOperator.__class__, IdentityOperator.__class__

    # TODO implement _series_expand

    @classmethod
    def create(cls, op):
        if isinstance(op, cls.delegate_to_method):
            return op._pseudo_inverse()
        return super(PseudoInverse, cls).create(op)

    @property
    def operand(self):
        return self.operands[0]

    _rules = []

    def _to_qutip(self, full_space=None):
        mo = self.operand.to_qutip(full_space)
        if full_space.dimension <= DENSE_DIMENSION_LIMIT:
            arr = mo.data.toarray()
            from scipy.linalg import pinv

            piarr = pinv(arr)
            pimo = qutip.Qobj(piarr)
            pimo.dims = mo.dims
            pimo.isherm = mo.isherm
            pimo.type = 'oper'
            return pimo
        raise NotImplementedError("Only implemented for smaller state spaces")

        #        return qutip.dag(self.operands[0].to_qutip(full_space))

    def _expand(self):
        return self

    def _pseudo_inverse(self):
        return self.operand

    def _tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^+"

    def __str__(self):
        if isinstance(self.operand, OperatorSymbol):
            return "{}^+".format(str(self.operand))
        return "({})^+".format(str(self.operand))


PseudoInverse.delegate_to_method = PseudoInverse.delegate_to_method + (PseudoInverse,)


@check_signature
@match_replace
class NullSpaceProjector(OperatorOperation):
    r"""
    Returns a projection operator :math:`\mathcal{P}_{{\rm Ker} X}` that projects onto the nullspace of its operand

    .. math::

        X \mathcal{P}_{{\rm Ker} X} = 0 \Leftrightarrow  X (1 - \mathcal{P}_{{\rm Ker} X}) = X\\
        \mathcal{P}_{{\rm Ker} X}^\dagger = \mathcal{P}_{{\rm Ker} X} = \mathcal{P}_{{\rm Ker} X}^2

    Use as:

        ``NullSpaceProjector(X)``

    :param X: Operator argument
    :type X: Operator
    """

    _rules = []

    # TODO implement _series_expand

    @property
    def operand(self):
        return self.operands[0]

    def to_qutip(self, full_space=None):
        mo = self.operand.to_qutip(full_space)
        if full_space.dimension <= DENSE_DIMENSION_LIMIT:
            arr = mo.data.toarray()
            from scipy.linalg import svd
            # compute Singular Value Decomposition
            U, s, Vh = svd(arr)
            tol = 1e-8 * s[0]
            zero_svs = s < tol
            Vhzero = Vh[zero_svs, :]
            PKarr = Vhzero.conjugate().transpose().dot(Vhzero)
            PKmo = qutip.Qobj(PKarr)
            PKmo.dims = mo.dims
            PKmo.isherm = True
            PKmo.type = 'oper'
            return PKmo
        raise NotImplementedError("Only implemented for smaller state spaces")


    def _tex(self):
        return r"\mathcal{P}_{{\rm Ker}" + tex(self.operand) + "}"

    def __str__(self):
        return "P_ker({})".format(str(self.operand))


@implied_local_space
@match_replace
@check_signature
class OperatorTrace(Operator, Operation):
    r"""
    Take the (partial) trace of an operator :math:`X` over the degrees of freedom given by a Hilbert space :math:`\mathcal{H}`:

    .. math::

        {\rm Tr}_{\mathcal{H}} X

    Use as:

        ``OperatorTrace(space, X)``

    :param space: The degrees of freedom to trace over
    :type space: HilbertSpace
    :param X: The operator to take the trace of.
    :type X: Operator
    """
    signature = HilbertSpace, Operator
    _rules = []

    def __init__(self, space, op):
        if isinstance(space, (int, str)):
            space = local_space(space)
        super(OperatorTrace, self).__init__(space, op)

    @property
    def _space(self):
        over_space, op = self.operands
        return op.space / over_space

    def _expand(self):
        s, o = self.operands
        return OperatorTrace.create(s, o.expand())

    def _series_expand(self, param, about, order):
        ope = self.operands[1].series_expand(param, about, order)
        return tuple(OperatorTrace.create(self.operands[0], opet) for opet in ope)

    def _tex(self):
        s, o = self.operands
        return r"{{\rm Tr}}_{{{}}} \left[ {} \right]".format(tex(s), tex(o))

    def __str__(self):
        s, o = self.operands
        return r"tr_{!s}[{!s}]".format(s, o)

    def _to_qutip(self, full_space):
        # TODO import OperatorTrace._to_qutip()
        raise NotImplementedError(self.__class__.__name__)

    def _all_symbols(self):
        return self.operands[1].all_symbols()


tr = OperatorTrace.create


def factor_for_trace(ls, op):
    r"""
    Given a local space ls to take the partial trace over and an operator, factor the trace such that operators acting on
    disjoint degrees of freedom are pulled out of the trace. If the operator acts trivially on ls the trace yields only
    a pre-factor equal to the dimension of ls. If there are LocalSigma operators among a product, the trace's cyclical property
    is used to move to sandwich the full product by :py:class:`LocalSigma` operators:

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
                return OperatorTimes.create(*pull_out) * OperatorTrace.create(ls, OperatorTimes.create(*rest))
        raise CannotSimplify()
    if ls & op.space == TrivialSpace:
        return ls.dimension * op
    if ls < op.space and isinstance(op, OperatorTimes):
        pull_out = [o for o in op.operands if (o.space & ls) == TrivialSpace]

        rest = [o for o in op.operands if (o.space & ls) != TrivialSpace]
        if not isinstance(rest[0], LocalSigma) or not isinstance(rest[-1], LocalSigma):
            found_ls = False
            for j, r in enumerate(rest):
                if isinstance(r, LocalSigma):
                    found_ls = True
                    break
            if found_ls:
                m, n = r.operands[1:]
                rest = rest[j:] + rest[:j] + [LocalSigma(ls, m, m)]
        if not rest:
            rest = [IdentityOperator]
        if len(pull_out):
            return OperatorTimes.create(*pull_out) * OperatorTrace.create(ls, OperatorTimes.create(*rest))
    raise CannotSimplify()


def decompose_space(H, A):
    """
    Simplifies OperatorTrace expressions over tensor-product spaces by turning it into iterated partial traces.
    :param H: The full space.
    :type H: ProductSpace
    :type A: Operator
    :return: Iterative partial trace expression
    :rtype: Operator
    """
    return OperatorTrace.create(ProductSpace.create(*H.operands[:-1]),
                                OperatorTrace.create(H.operands[-1], A))


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
    ((1, A), lambda A: A),
    ((0, A), lambda A: ZeroOperator),
    ((u, ZeroOperator), lambda u: ZeroOperator),
    ((u, ScalarTimesOperator(v, A)), lambda u, v, A: (u * v) * A)
]

OperatorPlus._binary_rules += [
    ((ScalarTimesOperator(u, A), ScalarTimesOperator(v, A)), lambda u, v, A: (u + v) * A),
    ((ScalarTimesOperator(u, A), A), lambda u, A: (u + 1) * A),
    ((A, ScalarTimesOperator(v, A)), lambda v, A: (1 + v) * A),
    ((A, A), lambda A: 2 * A),
]

OperatorTimes._binary_rules += [
    ((ScalarTimesOperator(u, A), B), lambda u, A, B: u * (A * B)),

    ((A, ScalarTimesOperator(u, B)), lambda A, u, B: u * (A * B)),

    ((LocalSigma(ls, ra, rb), LocalSigma(ls, rc, rd)),
     lambda ls, ra, rb, rc, rd: LocalSigma(ls, ra, rd)
     if rb == rc else ZeroOperator),

    ((Create(ls), LocalSigma(ls, rc, rd)),
     lambda ls, rc, rd: sqrt(rc + 1) * LocalSigma(ls, rc + 1, rd)),

    ((Destroy(ls), LocalSigma(ls, rc, rd)),
     lambda ls, rc, rd: sqrt(rc) * LocalSigma(ls, rc - 1, rd)),

    ((LocalSigma(ls, rc, rd), Destroy(ls)),
     lambda ls, rc, rd: sqrt(rd + 1) * LocalSigma(ls, rc, rd + 1)),

    ((LocalSigma(ls, rc, rd), Create(ls)),
     lambda ls, rc, rd: sqrt(rd) * LocalSigma(ls, rc, rd - 1)),

    ((Destroy(ls), Create(ls)),
     lambda ls: IdentityOperator + Create(ls) * Destroy(ls)),

    ((Phase(ls, u), Phase(ls, v)), lambda ls, u, v: Phase.create(ls, u + v)),
    ((Displace(ls, u), Displace(ls, v)),
     lambda ls, u, v: exp((u * v.conjugate() - u.conjugate() * v) / 2) * Displace.create(ls, u + v)),

    ((Destroy(ls), Phase(ls, u)), lambda ls, u: exp(I * u) * Phase(ls, u) * Destroy(ls)),
    ((Destroy(ls), Displace(ls, u)), lambda ls, u: Displace(ls, u) * (Destroy(ls) + u)),

    ((Phase(ls, u), Create(ls)), lambda ls, u: exp(I * u) * Create(ls) * Phase(ls, u)),
    ((Displace(ls, u), Create(ls)), lambda ls, u: (Create(ls) - u.conjugate()) * Displace(ls, u)),

    ((Phase(ls, u), LocalSigma(ls, n, m)), lambda ls, u, n, m: exp(I * u * n) * LocalSigma(ls, n, m)),
    ((LocalSigma(ls, n, m), Phase(ls, u)), lambda ls, u, n, m: exp(I * u * m) * LocalSigma(ls, n, m)),
]

Adjoint._rules += [
    ((ScalarTimesOperator(u, A),), lambda u, A: u.conjugate() * A.adjoint()),
    ((A_plus,), lambda A: OperatorPlus.create(*[o.adjoint() for o in A.operands])),
    ((A_times,), lambda A: OperatorTimes.create(*[o.adjoint() for o in A.operands[::-1]])),
    ((Adjoint(A),), lambda A: A),
    ((Create(ls),), lambda ls: Destroy(ls)),
    ((Destroy(ls),), lambda ls: Create(ls)),
    ((LocalSigma(ls, ra, rb),), lambda ls, ra, rb: LocalSigma(ls, rb, ra)),
]

Displace._rules += [
    ((ls, 0), lambda ls: IdentityOperator)
]
Phase._rules += [
    ((ls, 0), lambda ls: IdentityOperator)
]
Squeeze._rules += [
    ((ls, 0), lambda ls: IdentityOperator)
]

OperatorTrace._rules += [
    ((TrivialSpace, A), lambda A: A),
    ((h1, ZeroOperator), lambda h1: ZeroOperator),
    ((h1, IdentityOperator), lambda h1: h1.dimension * IdentityOperator),
    ((h1, A_plus), lambda h1, A: sum(OperatorTrace.create(h1, o) for o in A.operands)),
    ((h1, Adjoint(A)), lambda h1, A: Adjoint.create(OperatorTrace.create(h1, A))),
    ((h1, ScalarTimesOperator(u, A)), lambda h1, u, A: u * OperatorTrace.create(h1, A)),
    ((H_ProductSpace, A), lambda H, A: decompose_space(H, A)),
    ((ls, Create(ls)), lambda ls: ZeroOperator),
    ((ls, Destroy(ls)), lambda ls: ZeroOperator),
    ((ls, LocalSigma(ls, n, m)), lambda ls, n, m: IdentityOperator if n == m else ZeroOperator),
    ((ls, A), lambda ls, A: factor_for_trace(ls, A)),
]

PseudoInverse._rules += [
    (LocalSigma(ls, m, n), lambda ls, m, n: LocalSigma(ls, n, m)),
]


class NonSquareMatrix(Exception):
    pass


class Matrix(Expression):
    """
    Matrix with Operator (or scalar-) valued elements.

    """
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
        """
        The shape of the matrix ``(nrows, ncols)``

        :type: tuple
        """
        return self.matrix.shape

    @property
    def block_structure(self):
        """
        For square matrices this gives the block (-diagonal) structure of the matrix as a
        tuple of integers that sum up to the full dimension.

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


    def __hash__(self):
        if not self._hash:
            self._hash = hash((tuple(self.matrix.flatten()), self.matrix.shape, Matrix))
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Matrix) and (self.matrix == other.matrix).all()

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


    #    @_trace
    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented

    __truediv__ = __div__

    #    def __pow__(self, power):
    #        return OperatorMatrix(self.matrix.__pow__(power))

    def transpose(self):
        """
        :return: The transpose matrix
        :rtype: Matrix
        """
        return Matrix(self.matrix.T)

    def conjugate(self):
        """
        The element-wise conjugate matrix, i.e., if an element is an operator this means the adjoint operator,
        but no transposition of matrix elements takes place.
        :return: Element-wise hermitian conjugate matrix.
        :rtype: Matrix
        """
        return Matrix(np_conjugate(self.matrix))

    @property
    def T(self):
        """
        :return: Transpose matrix
        :rtype: Matrix
        """
        return self.transpose()

    def adjoint(self):
        """
        Return the adjoint operator matrix, i.e. transpose and the Hermitian adjoint operators of all elements.
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
        """
        Return the adjoint operator matrix, i.e. transpose and the Hermitian adjoint operators of all elements.
        """
        return self.adjoint()


    def __getitem__(self, item_id):
        item = self.matrix.__getitem__(item_id)
        if isinstance(item, ndarray):
            return Matrix(item)
        return item

    def element_wise(self, method):
        """
        Apply a method to each matrix element and return the result in a new operator matrix of the same shape.
        :param method: A method taking a single argument.
        :type method: FunctionType
        :return: Operator matrix with results of method applied element-wise.
        :rtype: Matrix
        """
        s = self.shape
        emat = [method(o) for o in self.matrix.flatten()]
        return Matrix(np_array(emat).reshape(s))

    def series_expand(self, param, about, order):
        """
        Expand the matrix expression as a truncated power series in a scalar parameter.

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
        emats = zip(*[o.series_expand(param, about, order) for o in self.matrix.flatten()])
        return tuple((Matrix(np_array(em).reshape(s)) for em in emats))


    def expand(self):
        """
        Expand each matrix element distributively.
        :return: Expanded matrix.
        :rtype: Matrix
        """
        m = lambda o: o.expand() if isinstance(o, Operator) else o
        return self.element_wise(m)

    def _substitute(self, var_map):

        def _substitute(o):
            sympy_var_map = {k: v for (k, v) in var_map.items() if isinstance(k, SympyBasic)}
            if isinstance(o, Operation):
                return substitute(o, var_map)
            elif isinstance(o, SympyBasic):
                return o.subs(sympy_var_map)
            else:
                return o

        return self.element_wise(_substitute)


    def _all_symbols(self):
        ret = set()
        for o in self.matrix.flatten():
            if isinstance(o, Operator):
                ret = ret | o.all_symbols()
            else:
                ret = ret | scalar_free_symbols(o)
        return ret


    def _tex(self):
        ret = r"\begin{pmatrix} "
        #        for row in self.matrix:
        ret += r""" \\ """.join([" & ".join([tex(o) for o in row]) for row in self.matrix])
        ret += r"\end{pmatrix}"

        return ret

    @property
    def space(self):
        """
        Return the combined Hilbert space of all matrix elements.

        :type: HilbertSpace
        """
        return prod((space(o) for o in self.matrix.flatten()), TrivialSpace)

    def simplify_scalar(self):
        """
        Simplify all scalar expressions appearing in the Matrix.
        """
        return self.element_wise(simplify_scalar)


def hstackm(matrices):
    """
    Generalizes `numpy.hstack` to OperatorMatrix objects.
    """
    return Matrix(np_hstack(tuple(m.matrix for m in matrices)))


def vstackm(matrices):
    """
    Generalizes `numpy.vstack` to OperatorMatrix objects.
    """
    arr = np_vstack(tuple(m.matrix for m in matrices))
    #    print(tuple(m.matrix.dtype for m in matrices))
    #    print(arr.dtype)
    return Matrix(arr)


def diagm(v, k=0):
    """
    Generalizes the diagonal matrix creation capabilities of `numpy.diag` to OperatorMatrix objects.
    """
    return Matrix(np_diag(v, k))


def block_matrix(A, B, C, D):
    r"""
    Generate the operator matrix with quadrants

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
    """
    Generate the N-dimensional identity matrix.

    :param N: Dimension
    :type N: int
    :return: Identity matrix in N dimensions
    :rtype: Matrix
    """
    return diagm(np_ones(N, dtype=int))


def zerosm(shape, *args, **kwargs):
    """
    Generalizes ``numpy.zeros`` to :py:class:`Matrix` objects.
    """
    return Matrix(np_zeros(shape, *args, **kwargs))


def permutation_matrix(permutation):
    r"""
    Return an orthogonal permutation matrix
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
    """
    The imaginary part of a number or operator. Acting on OperatorMatrices, it produces the element-wise imaginary parts.
    :param op: Anything that has a conjugate method.
    :type op: Operator or Matrix or any of Operator.scalar_types
    :return: The imaginary part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() - op) * I / 2


def Re(op):
    """
    The real part of a number or operator. Acting on OperatorMatrices, it produces the element-wise real parts.
    :param op: Anything that has a conjugate method.
    :type op: Operator or Matrix or any of Operator.scalar_types
    :return: The real part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() + op) / 2


def ImAdjoint(opmatrix):
    """
    The imaginary part of an OperatorMatrix, i.e. a hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: Matrix
    :return: The matrix imaginary part of the operand.
    :rtype: Matrix
    """
    return (opmatrix.H - opmatrix) * I / 2


def ReAdjoint(opmatrix):
    """
    The real part of an OperatorMatrix, i.e. a hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: Matrix
    :return: The matrix real part of the operand.
    :rtype: Matrix
    """
    return (opmatrix.H + opmatrix) / 2


def _coeff_term(expr):
    if isinstance(expr, ScalarTimesOperator):
        return expr.coeff, expr.term
    else:
        return 1, expr


def get_coeffs(expr, expand=False, epsilon=0.):
    """
    Create a dictionary with all Operator terms of the expression
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
    ret = defaultdict(float)
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


# class CannotFactorException(AlgebraException):
#     pass

# def factor_right(op1, op2, expand_op1=False, expand_op2=False):
#     """
#     Try to factor the operator op1 as X * op2.
#
#     :param op1: The full expression.
#     :type op1: Operator
#     :param op2: The operator to factor out.
#     :type op2: Operator
#     :param expand_op1: Whether to expand out op1 distributively.
#     :param expand_op2: Whether to expand out op2 distributively.
#     :return: The left factor X.
#     :rtype: Operator
#     """
#     if expand_op1:
#         op1 = op1.expand()
#     if expand_op2:
#         op2 = op2.expand()
#
#     if isinstance(op1, ScalarTimesOperator):
#         return op1.coeff * factor_right(op1.term, op2)
#
#     if isinstance(op2, ScalarTimesOperator):
#         return factor_right(op1, op2.term)/op2.coeff
#
#     if op1 is ZeroOperator:
#         return op1
#
#     if op2 is IdentityOperator:
#         return op1
#
#     if isinstance(op1, OperatorPlus):
#         if isinstance(op2, OperatorPlus):
#             A = op1
#             R = ZeroOperator
#             Qs = []
#             for Xj in op2.operands:
#                 Qj = ZeroOperator
#                 for Ak in A.operands:
#                     try:
#                         Qj += factor_right(Ak, Xj)
#                     except CannotFactorException:
#                         pass



