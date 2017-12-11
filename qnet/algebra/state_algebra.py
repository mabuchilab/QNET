# coding=utf-8
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

r"""This module implements the algebra of states in a Hilbert space

For more details see :ref:`state_algebra`.
"""
import re
import sympy
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import product as cartesian_product
from collections import OrderedDict

from sympy import (
    Basic as SympyBasic, series as sympy_series, sqrt, exp, I, factorial, Idx)

from .scalar_types import SCALAR_TYPES
from .abstract_algebra import (
    Operation, Expression, substitute, AlgebraError, assoc, orderby,
    filter_neutral, match_replace, match_replace_binary,
    CannotSimplify, check_rules_dict, InfiniteSumError, all_symbols)
from .singleton import Singleton, singleton_object
from .pattern_matching import wc, pattern_head, pattern
from .hilbert_space_algebra import (
    FullSpace, TrivialSpace, LocalSpace, ProductSpace, BasisNotSetError)
from .operator_algebra import (
    Operator, sympyOne, ScalarTimesOperator, OperatorTimes, OperatorPlus,
    IdentityOperator, ZeroOperator, LocalSigma, Create, Destroy, Jplus,
    Jminus, Jz, LocalOperator, Jpjmcoeff, Jzjmcoeff, Jmjmcoeff, Displace,
    Phase)
from .ordering import KeyTuple, expr_order_key, FullCommutativeHSOrder
from .indices import (
    SymbolicLabelBase, yield_from_ranges, IndexOverFockSpace, IndexOverRange,
    KroneckerDelta, IdxSym, FockIndex)

__all__ = [
    'OverlappingSpaces', 'SpaceTooLargeError', 'UnequalSpaces', 'BasisKet',
    'Bra', 'BraKet', 'CoherentStateKet', 'Ket', 'KetBra', 'KetPlus',
    'KetSymbol', 'LocalKet', 'OperatorTimesKet', 'ScalarTimesKet',
    'TensorKet', 'TrivialKet', 'ZeroKet', 'KetIndexedSum']

__private__ = [  # anything not in __all__ must be in __private__
    'act_locally', 'act_locally_times_tensor', 'tensor_decompose_kets',
    'check_kets_same_space', 'check_op_ket_space', 'accept_bras']


###############################################################################
# Exceptions
###############################################################################


class UnequalSpaces(AlgebraError):
    pass


class OverlappingSpaces(AlgebraError):
    pass


class SpaceTooLargeError(AlgebraError):
    pass


###############################################################################
# Algebraic properties
###############################################################################


def check_kets_same_space(cls, ops, kwargs):
    """Check that all operands are from the same Hilbert space."""
    if not all([isinstance(o, Ket) for o in ops]):
        raise TypeError("All operands must be Kets")
    if not len({o.space for o in ops if o is not ZeroKet}) == 1:
        raise UnequalSpaces(str(ops))
    return ops, kwargs


def check_op_ket_space(cls, ops, kwargs):
    """Check that all operands are from the same Hilbert space."""
    op, ket = ops
    if not op.space <= ket.space:
        raise SpaceTooLargeError(str(op.space) + " <!= " + str(ket.space))
    return ops, kwargs


def accept_bras(cls, ops, kwargs):
    """Accept operands that are all bras, and turn that into to bra of the
    operation applied to all corresponding kets"""
    kets = []
    for bra in ops:
        if isinstance(bra, Bra):
            kets.append(bra.ket)
        else:
            return ops, kwargs
    return Bra.create(cls.create(*kets, **kwargs))


###############################################################################
# Abstract base classes
###############################################################################


class Ket(metaclass=ABCMeta):
    """Basic Ket algebra class to represent Hilbert Space states"""

    @abstractproperty
    def space(self):
        """The associated HilbertSpace"""
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """The adjoint of a Ket state, i.e., the corresponding Bra."""
        return Bra(self)

    dag = property(adjoint)

    def expand(self):
        """Expand out distributively all products of sums of Kets."""
        return self._expand()

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

    def __add__(self, other):
        if isinstance(other, Ket):
            return KetPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return ScalarTimesKet.create(other, self)
        elif isinstance(other, Ket):
            return TensorKet.create(self, other)
        elif isinstance(other, Bra):
            return KetBra.create(self, other.ket)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return OperatorTimesKet.create(other, self)
        elif isinstance(other, SCALAR_TYPES):
            return ScalarTimesKet.create(other, self)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self * (sympyOne / other)
        return NotImplemented

    __truediv__ = __div__


###############################################################################
# Operator algebra elements
###############################################################################


class KetSymbol(Ket, Expression):
    """Symbolic state

    Args:
        label (str or SymbolicLabelBase): Symbol Identifier
        hs (HilbertSpace): associated Hilbert space (may be a
            :class:`~qnet.algebra.hilbert_space_algebra.ProductSpace`)
    """
    _rx_label = re.compile('^[A-Za-z0-9+-]+(_[A-Za-z0-9().,+-]+)?$')

    def __init__(self, label, *, hs):
        self._label = label
        if isinstance(label, str):
            if not self._rx_label.match(label):
                raise ValueError(
                    "label '%s' does not match pattern '%s'"
                    % (label, self._rx_label.pattern))
        elif isinstance(label, SymbolicLabelBase):
            self._label = label
        else:
            raise TypeError(
                "type of label must be str or SymbolicLabelBase, not %s"
                % type(label))
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        elif isinstance(hs, tuple):
            hs = ProductSpace.create(*[LocalSpace(h) for h in hs])
        self._hs = hs
        self._order_key = KeyTuple((self.__class__.__name__, str(label), 1.0))
        super().__init__(label, hs=hs)

    @property
    def args(self):
        return (self.label, )

    @property
    def kwargs(self):
        return {'hs': self._hs}

    @property
    def space(self):
        return self._hs

    @property
    def label(self):
        return self._label

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self, ) + (0, ) * (order - 1)

    def all_symbols(self):
        try:
            return self.label.all_symbols()
        except AttributeError:
            return set([self, ])


class LocalKet(KetSymbol):
    """A state that lives on a single local Hilbert space. This does
    not include operations, even if these operations only involve states acting
    on the same local space"""

    def __init__(self, label, *, hs):
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        if not isinstance(hs, LocalSpace):
            raise TypeError("hs must be a LocalSpace")
        super().__init__(label, hs=hs)

    def all_symbols(self):
        try:
            return self.label.all_symbols()
        except AttributeError:
            return set([])


@singleton_object
class ZeroKet(Ket, Expression, metaclass=Singleton):
    """ZeroKet constant (singleton) object for the null-state."""

    @property
    def space(self):
        return FullSpace

    @property
    def args(self):
        return tuple()

    @property
    def _order_key(self):
        return KeyTuple(('~~', self.__class__.__name__, 1.0))

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def __eq__(self, other):
        return self is other or other == 0

    def all_symbols(self):
        return set([])


@singleton_object
class TrivialKet(Ket, Expression, metaclass=Singleton):
    """TrivialKet constant (singleton) object.
    This is the neutral element under the state tensor-product.
    """

    @property
    def space(self):
        return TrivialSpace

    def _adjoint(self):
        return Bra(TrivialKet)

    def _expand(self):
        return self

    @property
    def args(self):
        return tuple()

    @property
    def _order_key(self):
        return KeyTuple(('~~', self.__class__.__name__, 1.0))

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def __eq__(self, other):
        return self is other or other == 1

    def all_symbols(self):
        return set([])


class BasisKet(LocalKet):
    """Local basis state, identified by index or label.

    Basis kets are orthornormal, and the :meth:`next` and :meth:`prev` methods
    can be used to move between basis states.

    Args:
        label_or_index: If `str`, the label of the basis state (must be an
            element of `hs.basis_labels`). If `int`, the (zero-based)
            index of the basis state. This works if `hs` has an unknown
            dimension. For a symbolic index, `label_or_index` can be an
            instance of an appropriate subclass of
            :class:`~qnet.algebra.indices.SymbolicLabelBase`
        hs (LocalSpace): The Hilbert space in which the basis is defined

    Raises:
        ValueError: if `label_or_index` is not in the Hilbert space
        TypeError: if `label_or_index` is not of an appropriate type
        ~qnet.algebra.hilbert_space_algebra.BasisNotSetError: if
            `label_or_index` is a `str` but no basis is defined for `hs`

    Note:
        Basis states that are instantiated via a label or via an index are
        equivalent::

            >>> hs = LocalSpace('tls', basis=('g', 'e'))
            >>> BasisKet('g', hs=hs) == BasisKet(0, hs=hs)
            True
            >>> print(ascii(BasisKet(0, hs=hs)))
            |g>^(tls)
    """
    def __init__(self, label_or_index, *, hs):
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        if not isinstance(hs, LocalSpace):
            raise TypeError("hs must be a LocalSpace")
        hs._check_basis_label_type(label_or_index)
        if isinstance(label_or_index, str):
            label = label_or_index
            ind = hs.basis_labels.index(label)  # raises BasisNotSetError
        elif isinstance(label_or_index, int):
            if hs.has_basis:
                label = hs.basis_labels[label_or_index]
            else:
                label = str(label_or_index)
            ind = label_or_index
            if ind < 0:
                raise ValueError("Index %d must be >= 0" % ind)
            if hs.has_basis:
                if ind >= hs.dimension:
                    raise ValueError(
                        "Index %s must be < the dimension %d of Hilbert "
                        "space %s" % (ind, hs.dimension, hs))
        elif isinstance(label_or_index, SymbolicLabelBase):
            label = label_or_index
            ind = label_or_index
        else:
            raise TypeError(
                "label_or_index must be an int or str, not %s"
                % type(label_or_index))
        self._index = ind
        super().__init__(label, hs=hs)

    @property
    def args(self):
        """Tuple containing `label_or_index` as its only element."""
        if self.space.has_basis:
            return (self.label, )
        else:
            return (self.index, )

    @property
    def index(self):
        """The index of the state in the Hilbert space basis

        >>> hs =  LocalSpace('tls', basis=('g', 'e'))
        >>> BasisKet('g', hs=hs).index
        0
        >>> BasisKet('e', hs=hs).index
        1
        >>> BasisKet(1, hs=hs).index
        1
        """
        return self._index

    def next(self, n=1):
        """Move up by `n` steps in the Hilbert space::

            >>> hs =  LocalSpace('tls', basis=('g', 'e'))
            >>> ascii(BasisKet('g', hs=hs).next())
            '|e>^(tls)'
            >>> ascii(BasisKet(0, hs=hs).next())
            '|e>^(tls)'

        We can also go multiple steps:

            >>> hs =  LocalSpace('ten', dimension=10)
            >>> ascii(BasisKet(0, hs=hs).next(2))
            '|2>^(ten)'

        An increment that leads out of the Hilbert space returns zero::

            >>> BasisKet(0, hs=hs).next(10)
            ZeroKet

        """
        try:
            next_index = self.space.next_basis_label_or_index(self.index, n)
            return BasisKet(next_index, hs=self.space)
        except IndexError:
            return ZeroKet

    def prev(self, n=1):
        """Move down by `n` steps in the Hilbert space, cf. :meth:`next`.

        >>> hs =  LocalSpace('3l', basis=('g', 'e', 'r'))
        >>> ascii(BasisKet('r', hs=hs).prev(2))
        '|g>^(3l)'
        >>> BasisKet('r', hs=hs).prev(3)
        ZeroKet
        """
        return self.next(n=-n)


class CoherentStateKet(LocalKet):
    """Local coherent state, labeled by a scalar amplitude.

    :param LocalSpace hs: The local Hilbert space degree of freedom.
    :param SCALAR_TYPES amp: The coherent displacement amplitude.
    """
    _rx_label = re.compile('^.*$')

    def __init__(self, ampl, *, hs):
        self._ampl = ampl
        label = 'alpha=' + str(hash(ampl))
        # The "label" here is something that is solely used to set the
        # _instance_key / uniquely identify the instance. Using the hash gets
        # around variations in str(ampl) due to printer settings
        # TODO: is this safe against hash collisions?
        super().__init__(label, hs=hs)

    @property
    def ampl(self):
        return self._ampl

    @property
    def args(self):
        return (self._ampl, )

    @property
    def label(self):
        return 'alpha'

    @property
    def kwargs(self):
        return {'hs': self._hs}

    def _series_expand(self, param, about, order):
        return (self,) + (0,) * (order - 1)

    def _substitute(self, var_map):
        hs, amp = self.space, self._ampl
        if isinstance(amp, SympyBasic):
            svar_map = {k: v for (k, v) in var_map.items()
                        if not isinstance(k, Expression)}
            ampc = amp.subs(svar_map)
        else:
            ampc = substitute(amp, var_map)

        return CoherentStateKet(ampc, hs=hs)

    def all_symbols(self):
        if isinstance(self.ampl, SympyBasic):
            return set([self.ampl, ])
        else:
            return set([])

    def to_fock_representation(self, index_symbol='n', max_terms=None):
        """Return the coherent state written out as an indexed sum over Fock
        basis states"""
        phase_factor = sympy.exp(
            sympy.Rational(-1, 2) * self.ampl * self.ampl.conjugate())
        if not isinstance(index_symbol, IdxSym):
            index_symbol = IdxSym(index_symbol)
        n = index_symbol
        if max_terms is None:
            index_range = IndexOverFockSpace(n, hs=self._hs)
        else:
            index_range = IndexOverRange(n, 0, max_terms-1)
        term = (
            (self.ampl**n / sympy.sqrt(sympy.factorial(n))) *
            BasisKet(FockIndex(n), hs=self._hs))
        return phase_factor * KetIndexedSum(term, index_range)


###############################################################################
# Algebra Operations
###############################################################################


class KetPlus(Ket, Operation):
    """A sum of Ket states.

    Instantiate as::

        KetPlus(*summands)

    :param summands: State summands.
    :type summands: Ket
    """
    neutral_element = ZeroKet
    _binary_rules = OrderedDict()  # see end of module
    _simplifications = [
        accept_bras, assoc, orderby, filter_neutral, check_kets_same_space,
        match_replace_binary]

    order_key = FullCommutativeHSOrder

    def __init__(self, *operands):
        self._order_key = KeyTuple([
            '~'+self.__class__.__name__,
            ", ".join([op.__class__.__name__ for op in operands]),
            1.0, tuple([op._order_key for op in operands])])
        super().__init__(*operands)

    @property
    def space(self):
        return self.operands[0].space

    def _expand(self):
        return KetPlus.create(*[o.expand() for o in self.operands])

    def _series_expand(self, param, about, order):
        return KetPlus.create(*[o.series_expand(param, about, order)
                                for o in self.operands])


class TensorKet(Ket, Operation):
    """A tensor product of kets each belonging to different degrees of freedom.
    Instantiate as::

        TensorKet(*factors)

    :param factors: Ket factors.
    :type factors: Ket
    """
    _binary_rules = OrderedDict()  # see end of module
    neutral_element = TrivialKet
    _simplifications = [
        accept_bras, assoc, orderby, filter_neutral, match_replace_binary]

    order_key = FullCommutativeHSOrder

    def __init__(self, *operands):
        self._space = None
        self._label = None
        if all(isinstance(o, LocalKet) for o in operands):
            self._label = ",".join([str(o.label) for o in operands])
        self._order_key = KeyTuple([
            '~'+self.__class__.__name__,
            ", ".join([op.__class__.__name__ for op in operands]),
            1.0, tuple([op._order_key for op in operands])])
        super().__init__(*operands)

    @classmethod
    def create(cls, *ops):
        if any(o == ZeroKet for o in ops):
            return ZeroKet
        spc = TrivialSpace
        for o in ops:
            if o.space & spc > TrivialSpace:
                raise OverlappingSpaces(str(ops))
            spc *= o.space
        return super().create(*ops)

    def factor_for_space(self, space):
        """Factor into a Ket defined on the given `space` and a Ket on the
        remaining Hilbert space"""
        if not space <= self.space:
            raise SpaceTooLargeError(str((self, space)))
        if space == self.space:
            return self
        if space is TrivialSpace:
            on_ops = [o for o in self.operands if o.space is TrivialSpace]
            off_ops = [o for o in self.operands if o.space > TrivialSpace]
        else:
            on_ops = [o for o in self.operands
                      if o.space & space > TrivialSpace]
            off_ops = [o for o in self.operands
                       if o.space & space is TrivialSpace]
        return TensorKet.create(*on_ops), TensorKet.create(*off_ops)

    @property
    def space(self):
        if self._space is None:
            self._space = ProductSpace.create(*[o.space
                                                for o in self.operands])
        return self._space

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, KetPlus) else (eo,)
                        for eo in eops]
        # iterate over a Cartesian product of all factor summands, form product
        # of each tuple and sum over result
        return KetPlus.create(
                *[TensorKet.create(*combo)
                  for combo in cartesian_product(*eopssummands)])

    @property
    def label(self):
        """Combined label of the product state if the state is a simple product
        of LocalKets, raise AttributeError otherwise"""
        if self._label is None:
            raise AttributeError("%r instance has no defined 'label'" % self)
        else:
            return self._label


class ScalarTimesKet(Ket, Operation):
    """Multiply a Ket by a scalar coefficient.

    Args:
        coeff (SCALAR_TYPES): coefficient
        term (Ket): the ket that is multiplied
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, ]

    def __init__(self, coeff, term):
        super().__init__(coeff, term)

    @property
    def _order_key(self):
        from qnet.printing import ascii
        t = self.term._order_key
        try:
            c = abs(float(self.coeff))  # smallest coefficients first
        except (ValueError, TypeError):
            c = float('inf')
        return KeyTuple(t[:2] + (c, ) + t[3:] + (ascii(self.coeff), ))

    @property
    def space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def _expand(self):
        c, t = self.coeff, self.term
        et = t.expand()
        if isinstance(et, KetPlus):
            return KetPlus.create(*[c * eto for eto in et.operands])
        return c * et

    def _series_expand(self, param, about, order):
        ceg = sympy_series(self.coeff, x=param, x0=about, n=None)
        ce = tuple(ceo for ceo, k in zip(ceg, range(order + 1)))
        te = self.term.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k]
                     for n in range(order + 1) for k in range(n + 1))

    def _substitute(self, var_map):
        st = self.term.substitute(var_map)
        if isinstance(self.coeff, SympyBasic):
            svar_map = {k: v for (k, v) in var_map.items()
                        if not isinstance(k, Expression)}
            sc = self.coeff.subs(svar_map)
        else:
            sc = substitute(self.coeff, var_map)
        return sc * st


class OperatorTimesKet(Ket, Operation):
    """Multiply an operator by an operator

    Instantiate as::

        OperatorTimesKet(op, ket)

    :param Operator op: The multiplying operator.
    :param Ket ket: The ket that is multiplied.
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, check_op_ket_space]

    @property
    def _order_key(self):
        return KeyTuple(
            (self.__class__.__name__, self.ket.__class__.__name__, 1.0) +
            self.ket._order_key + self.operator._order_key)

    @property
    def space(self):
        return self.operands[1].space

    @property
    def operator(self):
        return self.operands[0]

    @property
    def ket(self):
        return self.operands[1]

    def _expand(self):
        c, t = self.operands
        ct = c.expand()
        et = t.expand()
        if isinstance(et, KetPlus):
            if isinstance(ct, OperatorPlus):
                return sum((cto * eto
                            for eto in et.operands
                            for cto in ct.operands), ZeroKet)
            else:
                return sum((c * eto for eto in et.operands), ZeroKet)
        elif isinstance(ct, OperatorPlus):
            return sum((cto * et for cto in ct.operands), ZeroKet)
        return ct * et

    def _series_expand(self, param, about, order):
        ce = self.coeff.series_expand(param, about, order)
        te = self.term.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class Bra(Operation):
    """The associated dual/adjoint state for any ``Ket`` object ``k`` is given
    by ``Bra(k)``.

    :param Ket k: The state to represent as Bra.
    """
    def __init__(self, ket):
        self._order_key = KeyTuple(
                (self.__class__.__name__, ket.__class__.__name__, 1.0) +
                ket._order_key)
        super().__init__(ket)

    @property
    def ket(self):
        """The state that is represented as a Bra.

        :rtype: Ket
        """
        return self.operands[0]

    operand = ket

    def adjoint(self):
        """The adjoint of a ``Bra`` is just the original ``Ket`` again."""
        return self.ket

    dag = property(adjoint)

    def expand(self):
        """Expand out distributively all products of sums of Bras."""
        return Bra(self.ket.expand())

    @property
    def space(self):
        return self.operands[0].space

    @property
    def label(self):
        return self.ket.label

    def __mul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return Bra.create(self.ket * other.conjugate())
        elif isinstance(other, Operator):
            return Bra.create(other.adjoint() * self.ket)
        elif isinstance(other, Ket):
            return BraKet.create(self.ket, other)
        elif isinstance(other, Bra):
            return Bra.create(self.ket * other.ket)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return Bra.create(self.ket * other.conjugate())
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Bra):
            return Bra.create(self.ket + other.ket)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Bra):
            return Bra.create(self.ket - other.ket)
        return NotImplemented

    def __div__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return Bra.create(self.ket/other.conjugate())
        return NotImplemented
    __truediv__ = __div__


class BraKet(Operator, Operation):
    r"""The symbolic inner product between two states, represented as Bra and
    Ket

    In math notation this corresponds to:

    .. math::
        \langle b | k \rangle

    which we define to be linear in the state :math:`k` and anti-linear in
    :math:`b`.

    :param Ket bra: The anti-linear state argument.
    :param Ket ket: The linear state argument.
    """
    _rules = OrderedDict()  # see end of module
    _space = TrivialSpace
    _simplifications = [check_kets_same_space, match_replace]

    def __init__(self, bra, ket):
        self._order_key = KeyTuple((self.__class__.__name__, '__', 1.0,
                                    expr_order_key(bra), expr_order_key(ket)))
        super().__init__(bra, ket)

    @property
    def ket(self):
        return self.operands[1]

    @property
    def bra(self):
        return Bra(self.operands[0])

    @property
    def space(self):
        return TrivialSpace

    def _adjoint(self):
        return BraKet.create(*reversed(self.operands))

    def _expand(self):
        b, k = self.operands
        be, ke = b.expand(), k.expand()
        besummands = be.operands if isinstance(be, KetPlus) else (be,)
        kesummands = ke.operands if isinstance(ke, KetPlus) else (ke,)
        return sum(BraKet.create(bes, kes)
                   for bes in besummands for kes in kesummands)

    def _series_expand(self, param, about, order):
        be = self.bra.series_expand(param, about, order)
        ke = self.ket.series_expand(param, about, order)
        return tuple(be[k] * ke[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class KetBra(Operator, Operation):
    """A symbolic operator formed by the outer product of two states

    :param Ket ket: The first state that defines the range of the operator.
    :param Ket bra: The second state that defines the Kernel of the operator.
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [check_kets_same_space, match_replace]

    def __init__(self, ket, bra):
        self._order_key = KeyTuple((self.__class__.__name__, '__', 1.0,
                                    expr_order_key(ket), expr_order_key(bra)))
        super().__init__(ket, bra)

    @property
    def ket(self):
        return self.operands[0]

    @property
    def bra(self):
        return Bra(self.operands[1])

    @property
    def space(self):
        return self.operands[0].space

    def _adjoint(self):
        return KetBra.create(*reversed(self.operands))

    def _expand(self):
        k, b = self.ket, self.bra.ket
        be, ke = b.expand(), k.expand()
        kesummands = ke.operands if isinstance(ke, KetPlus) else (ke,)
        besummands = be.operands if isinstance(be, KetPlus) else (be,)
        res_summands = []
        for (k, b) in cartesian_product(kesummands, besummands):
            res_summands.append(KetBra.create(k, b))
        return OperatorPlus.create(*res_summands)

    def _series_expand(self, param, about, order):
        ke = self.ket.series_expand(param, about, order)
        be = self.bra.series_expand(param, about, order)
        return tuple(ke[k] * be[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class KetIndexedSum(Ket, Operation):
    # TODO: documentation

    _expanded_cls = KetPlus
    _expand_limit = 1000

    def __init__(self, term, *ranges):
        self._term = term
        self.ranges = tuple(ranges)
        super().__init__(term, ranges=ranges)

    @property
    def space(self):
        return self.term.space

    @property
    def term(self):
        return self._term

    @property
    def operands(self):
        return (self._term, )

    @property
    def args(self):
        return tuple([self._term, *self.ranges])

    @property
    def variables(self):
        """List of the dummy (index) variable symbols"""
        return [r.index_symbol for r in self.ranges]

    def all_symbols(self):
        """Set of all free symbols"""
        return set(
            [sym for sym in all_symbols(self.term)
                if sym not in self.variables])

    @property
    def kwargs(self):
        return {}

    def _expand(self):
        return self.__class__.create(self.term.expand(), *self.ranges)

    def _series_expand(self, param, about, order):
        raise NotImplementedError()

    @property
    def terms(self):
        for mapping in yield_from_ranges(self.ranges):
            yield self.term.substitute(mapping).simplify(rules=[(
                wc('label', head=SymbolicLabelBase),
                lambda label: label.evaluate(mapping))])

    def __len__(self):
        length = 1
        try:
            for ind_range in self.ranges:
                length *= len(ind_range)
        except BasisNotSetError:
            raise InfiniteSumError(
                "sum %s has infinite length" % self)
        return length

    def expand_sum(self, max_terms=None):
        terms = []
        if max_terms is None:
            len(self)  # side-effect: raise InfiniteSumError
        else:
            if max_terms > self._expand_limit:
                raise ValueError(
                    "max_terms = %s must be smaller than the limit %s"
                    % (max_terms, self._expand_limit))
        for i, term in enumerate(self.terms):
            if max_terms is not None:
                if i >= max_terms:
                    break
            terms.append(term)
            if i > self._expand_limit:
                raise InfiniteSumError(
                    "Cannot expand %s: more than %s terms"
                    % (self, self._expand_limit))
        return self._expanded_cls.create(*terms)


###############################################################################
# Auxilliary routines
###############################################################################


def act_locally(op, ket):
    ket_on, ket_off = ket.factor_for_space(op.space)
    if ket_off != TrivialKet:
        return (op * ket_on) * ket_off
    raise CannotSimplify()


def act_locally_times_tensor(op, ket):
    local_spaces = op.space.local_factors
    for spc in local_spaces:
        while spc < ket.space:
            op_on, op_off = op.factor_for_space(spc)
            ket_on, ket_off = ket.factor_for_space(spc)

            if (op_on.space <= ket_on.space and
                    op_off.space <= ket_off.space and ket_off != TrivialKet):
                return (op_on * ket_on) * (op_off * ket_off)
            else:
                spc = op_on.space * ket_on.space
    raise CannotSimplify()


def tensor_decompose_kets(a, b, operation):
    full_space = a.space * b.space
    local_spaces = full_space.local_factors
    for spc in local_spaces:
        while spc < full_space:
            a_on, a_off = a.factor_for_space(spc)
            b_on, b_off = b.factor_for_space(spc)
            if (a_on.space == b_on.space and a_off.space == b_off.space and
                    a_off != TrivialKet):
                return operation(a_on, b_on) * operation(a_off, b_off)
            else:
                spc = a_on.space * b_on.space
    raise CannotSimplify()


###############################################################################
# Algebraic rules
###############################################################################


def _algebraic_rules():
    """Set the default algebraic rules for the operations defined in this
    module"""
    u = wc("u", head=SCALAR_TYPES)
    v = wc("v", head=SCALAR_TYPES)

    n = wc("n", head=(int, str, SymbolicLabelBase))
    m = wc("m", head=(int, str, SymbolicLabelBase))
    k = wc("k", head=(int, str, SymbolicLabelBase))

    A = wc("A", head=Operator)
    A__ = wc("A__", head=Operator)
    B = wc("B", head=Operator)

    A_times = wc("A", head=OperatorTimes)
    A_local = wc("A", head=LocalOperator)
    B_local = wc("B", head=LocalOperator)

    nsym = wc("nsym", head=(int, str, SympyBasic))

    Psi = wc("Psi", head=Ket)
    Phi = wc("Phi", head=Ket)
    Psi_local = wc("Psi", head=LocalKet)
    Psi_tensor = wc("Psi", head=TensorKet)
    Phi_tensor = wc("Phi", head=TensorKet)

    ls = wc("ls", head=LocalSpace)

    basisket = wc('basisket', BasisKet, kwargs={'hs': ls})

    ScalarTimesKet._rules.update(check_rules_dict([
        ('one', (
            pattern_head(1, Psi),
            lambda Psi: Psi)),
        ('zero', (
            pattern_head(0, Psi),
            lambda Psi: ZeroKet)),
        ('uzero', (
            pattern_head(u, ZeroKet),
            lambda u: ZeroKet)),
        ('uvpsi', (
            pattern_head(u, pattern(ScalarTimesKet, v, Psi)),
            lambda u, v, Psi: (u * v) * Psi))
    ]))

    # local_rule = lambda A, B, Psi: OperatorTimes.create(*A) * (B * Psi)

    def local_rule(A, B, Psi):
        return OperatorTimes.create(*A) * (B * Psi)

    OperatorTimesKet._rules.update(check_rules_dict([
        ('id', (
            pattern_head(IdentityOperator, Psi),
            lambda Psi: Psi)),
        ('zeroop', (
            pattern_head(ZeroOperator, Psi),
            lambda Psi: ZeroKet)),
        ('zeroket', (
            pattern_head(A, ZeroKet),
            lambda A: ZeroKet)),
        ('vApsi', (
            pattern_head(A, pattern(ScalarTimesKet, v, Psi)),
            lambda A, v, Psi:  v * (A * Psi))),

        ('sig', (
            pattern_head(
                pattern(LocalSigma, n, m, hs=ls),
                pattern(BasisKet, k, hs=ls)),
            lambda ls, n, m, k: KroneckerDelta(m, k) * BasisKet(n, hs=ls))),

        # harmonic oscillator
        ('create', (
            pattern_head(pattern(Create, hs=ls), basisket),
            lambda basisket, ls:
                sqrt(basisket.index + 1) * basisket.next())),
        ('destroy', (
            pattern_head(pattern(Destroy, hs=ls), basisket),
            lambda basisket, ls:
                sqrt(basisket.index) * basisket.prev())),
        ('coherent', (
            pattern_head(
                pattern(Destroy, hs=ls),
                pattern(CoherentStateKet, u, hs=ls)),
            lambda ls, u: u * CoherentStateKet(u, hs=ls))),

        # spin
        ('jplus', (
            pattern_head(pattern(Jplus, hs=ls), basisket),
            lambda basisket, ls:
                Jpjmcoeff(basisket.space, basisket.index, shift=True) *
                basisket.next())),
        ('jminus', (
            pattern_head(pattern(Jminus, hs=ls), basisket),
            lambda basisket, ls:
                Jmjmcoeff(basisket.space, basisket.index, shift=True) *
                basisket.prev())),
        ('jz', (
            pattern_head(pattern(Jz, hs=ls), basisket),
            lambda basisket, ls:
                Jzjmcoeff(basisket.space, basisket.index, shift=True) *
                basisket)),

        ('local1', (
            pattern_head(A_local, Psi_tensor),
            lambda A, Psi: act_locally(A, Psi))),
        ('prod', (
            pattern_head(A_times, Psi_tensor),
            lambda A, Psi: act_locally_times_tensor(A, Psi))),
        ('ABPsi', (
            pattern_head(A, pattern(OperatorTimesKet, B, Psi)),
            lambda A, B, Psi: (
                (A * B) * Psi
                if (B * Psi) == OperatorTimesKet(B, Psi)
                else A * (B * Psi)))),
        ('local2', (
            pattern_head(pattern(OperatorTimes, A__, B_local), Psi_local),
            local_rule)),
        ('uAPsi', (
            pattern_head(pattern(ScalarTimesOperator, u, A), Psi),
            lambda u, A, Psi: u * (A * Psi))),
        ('displace1', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(BasisKet, 0, hs=ls)),
            lambda ls, u: CoherentStateKet(u, hs=ls))),
        ('displace2', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(CoherentStateKet, v, hs=ls)),
            lambda ls, u, v:
                ((Displace(u, hs=ls) * Displace(v, hs=ls)) *
                 BasisKet(0, hs=ls)))),
        ('phase1', (
            pattern_head(
                pattern(Phase, u, hs=ls), pattern(BasisKet, m, hs=ls)),
            lambda ls, u, m: exp(I * u * m) * BasisKet(m, hs=ls))),
        ('phase2', (
            pattern_head(
                pattern(Phase, u, hs=ls),
                pattern(CoherentStateKet, v, hs=ls)),
            lambda ls, u, v: CoherentStateKet(v * exp(I * u), hs=ls))),
    ]))

    KetPlus._binary_rules.update(check_rules_dict([
        ('scalsum', (
            pattern_head(
                pattern(ScalarTimesKet, u, Psi),
                pattern(ScalarTimesKet, v, Psi)),
            lambda u, v, Psi: (u + v) * Psi)),
        ('up1', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Psi),
            lambda u, Psi: (u + 1) * Psi)),
        ('1pv', (
            pattern_head(Psi, pattern(ScalarTimesKet, v, Psi)),
            lambda v, Psi: (1 + v) * Psi)),
        ('2psi', (
            pattern_head(Psi, Psi),
            lambda Psi: 2 * Psi)),
    ]))

    TensorKet._binary_rules.update(check_rules_dict([
        ('scal1', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
            lambda u, Psi, Phi: u * (Psi * Phi))),
        ('scal2', (
            pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
            lambda Psi, u, Phi: u * (Psi * Phi))),
    ]))

    BraKet._rules.update(check_rules_dict([
        ('zero2', (
            pattern_head(Phi, ZeroKet),
            lambda Phi: 0)),
        ('zero1', (
            pattern_head(ZeroKet, Phi),
            lambda Phi: 0)),
        ('dirac', (
            pattern_head(
                pattern(BasisKet, m, hs=ls), pattern(BasisKet, n, hs=ls)),
            lambda ls, m, n: KroneckerDelta(m, n))),
        ('norm', (
            pattern_head(
                pattern(BasisKet, nsym, hs=ls),
                pattern(BasisKet, nsym, hs=ls)),
            lambda ls, nsym: 1)),
        ('tensor', (
            pattern_head(Psi_tensor, Phi_tensor),
            lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, BraKet.create))),
        ('scal1', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
            lambda u, Psi, Phi: u.conjugate() * (Psi.adjoint() * Phi))),
        ('op', (
            pattern_head(pattern(OperatorTimesKet, A, Psi), Phi),
            lambda A, Psi, Phi: (Psi.adjoint() * (A.dag() * Phi)))),
        ('scal2', (
            pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
            lambda Psi, u, Phi: u * (Psi.adjoint() * Phi))),
    ]))

    KetBra._rules.update(check_rules_dict([
        ('sig', (
            pattern_head(
                pattern(BasisKet, m, hs=ls),
                pattern(BasisKet, n, hs=ls)),
            lambda ls, m, n: LocalSigma(m, n, hs=ls))),
        ('coherent1', (
            pattern_head(pattern(CoherentStateKet, u, hs=ls), Phi),
            lambda ls, u, Phi: (
                Displace(u, hs=ls) * (BasisKet(0, hs=ls) * Phi.adjoint())))),
        ('coherent2', (
            pattern_head(Phi, pattern(CoherentStateKet, u, hs=ls)),
            lambda ls, u, Phi: (
                (Phi * BasisKet(0, hs=ls).adjoint()) * Displace(-u, hs=ls)))),
        ('tensor', (
            pattern_head(Psi_tensor, Phi_tensor),
            lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, KetBra.create))),
        ('op1', (
            pattern_head(pattern(OperatorTimesKet, A, Psi), Phi),
            lambda A, Psi, Phi: A * (Psi * Phi.adjoint()))),
        ('op2', (
            pattern_head(Psi, pattern(OperatorTimesKet, A, Phi)),
            lambda Psi, A, Phi: (Psi * Phi.adjoint()) * A.adjoint())),
        ('scal1', (
            pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
            lambda u, Psi, Phi: u * (Psi * Phi.adjoint()))),
        ('scal2', (
            pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
            lambda Psi, u, Phi: u.conjugate() * (Psi * Phi.adjoint()))),
    ]))


_algebraic_rules()
