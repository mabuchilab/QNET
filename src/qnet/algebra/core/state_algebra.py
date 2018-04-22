r"""This module implements the algebra of states in a Hilbert space

For more details see :ref:`state_algebra`.
"""
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product as cartesian_product

import sympy
from sympy import (
    Basic as SympyBasic, series as sympy_series, )

from .abstract_algebra import (
    Expression, Operation, ScalarTimesExpression, substitute, )
from .algebraic_properties import (
    accept_bras, assoc, assoc_indexed, basis_ket_zero_outside_hs,
    check_kets_same_space, check_op_ket_space, filter_neutral, match_replace,
    match_replace_binary, orderby, )
from .exceptions import (OverlappingSpaces, SpaceTooLargeError)
from .hilbert_space_algebra import (
    FullSpace, LocalSpace, ProductSpace, TrivialSpace, )
from .indexed_operations import (
    IndexedSum, indexed_sum_over_const, indexed_sum_over_kronecker, )
from .operator_algebra import (
    Operator, OperatorIndexedSum, OperatorPlus, sympyOne, )
from .scalar_types import SCALAR_TYPES
from ...utils.indices import (
    FockIndex, IdxSym, IndexOverFockSpace, IndexOverRange, SymbolicLabelBase, )
from ...utils.ordering import FullCommutativeHSOrder, KeyTuple, expr_order_key
from ...utils.singleton import Singleton, singleton_object

__all__ = [
    'BasisKet', 'Bra', 'BraKet', 'CoherentStateKet', 'Ket', 'KetBra',
    'KetPlus', 'KetSymbol', 'LocalKet', 'OperatorTimesKet', 'ScalarTimesKet',
    'TensorKet', 'TrivialKet', 'ZeroKet', 'KetIndexedSum']

__private__ = []  # anything not in __all__ must be in __private__


###############################################################################
# Algebraic properties
###############################################################################


class Ket(metaclass=ABCMeta):
    """Basic Ket algebra class to represent Hilbert Space states"""

    @property
    @abstractmethod
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
            if isinstance(other, KetIndexedSum):
                return other.__class__.create(self * other.term, *other.ranges)
            else:
                return TensorKet.create(self, other)
        elif isinstance(other, Bra):
            if isinstance(other.ket, KetIndexedSum):
                return OperatorIndexedSum.create(
                    self * other.ket.term.dag, *other.ket.ranges)
            else:
                return KetBra.create(self, other.ket)
        try:
            return super().__mul__(other)
        except AttributeError:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return OperatorTimesKet.create(other, self)
        elif isinstance(other, SCALAR_TYPES):
            return ScalarTimesKet.create(other, self)
        try:
            return super().__rmul__(other)
        except AttributeError:
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
        try:
            return super().__div__(other)
        except AttributeError:
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
            return {self}


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
        .BasisNotSetError: if `label_or_index` is a `str` but no basis is
            defined for `hs`

    Note:
        Basis states that are instantiated via a label or via an index are
        equivalent::

            >>> hs = LocalSpace('tls', basis=('g', 'e'))
            >>> BasisKet('g', hs=hs) == BasisKet(0, hs=hs)
            True
            >>> print(ascii(BasisKet(0, hs=hs)))
            |g>^(tls)

        When instantiating the :class:`BasisKet` via
        :meth:`~qnet.algebra.abstract_algebra.Expression.create`, an integer
        label outside the range of the underlying Hilbert space results in a
        :class:`ZeroKet`::

            >>> BasisKet.create(-1, hs=0)
            ZeroKet
            >>> BasisKet.create(2, hs=LocalSpace('tls', dimension=2))
            ZeroKet
    """

    _simplifications = [basis_ket_zero_outside_hs]

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
            return {self.ampl}
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
    """A sum of states."""
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


class ScalarTimesKet(Ket, ScalarTimesExpression):
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

        return tuple(
            ce[k] * te[n - k]
            for n in range(order + 1) for k in range(n + 1))


class OperatorTimesKet(Ket, Operation):
    """Product of an operator and a state."""
    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, check_op_ket_space]

    def __init__(self, operator, ket):
        super().__init__(operator, ket)

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
    """The associated dual/adjoint state for any `ket`"""
    def __init__(self, ket):
        self._order_key = KeyTuple(
                (self.__class__.__name__, ket.__class__.__name__, 1.0) +
                ket._order_key)
        super().__init__(ket)

    @property
    def ket(self):
        """The original :class:`Ket`"""
        return self.operands[0]

    operand = ket

    def adjoint(self):
        """The adjoint of a :class:`Bra` is just the original :class:`Ket`
        again"""
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
        if isinstance(self.ket, KetIndexedSum):
            if isinstance(other, KetIndexedSum):
                other = other.make_disjunct_indices(self.ket)
                assert isinstance(other, KetIndexedSum)
                new_ranges = self.ket.ranges + other.ranges
                new_term = BraKet.create(self.ket.term, other.term)
                return OperatorIndexedSum.create(new_term, *new_ranges)
            elif isinstance(other, Ket):
                return OperatorIndexedSum(
                    BraKet.create(self.ket.term, other), *self.ket.ranges)
        if isinstance(other, SCALAR_TYPES):
            return Bra.create(self.ket * other.conjugate())
        elif isinstance(other, Operator):
            return Bra.create(other.adjoint() * self.ket)
        elif isinstance(other, Ket):
            if isinstance(other, KetIndexedSum):
                return OperatorIndexedSum(
                    BraKet.create(self.ket, other.term), *other.ranges)
            else:
                return BraKet.create(self.ket, other)
        elif isinstance(other, Bra):
            return Bra.create(self.ket * other.ket)
        try:
            return super().__mul__(other)
        except AttributeError:
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

    Args:
        bra (Ket): The anti-linear state argument. Not that this is *not* a
            :class:`Bra` instance.
        ket (Ket): The linear state argument.

    Note:
        :class:`Braket` is an :class:`Operator` in the
        :class:`~qnet.algebra.hilbert_space_algebra.TrivialSpace`, that is,
        ultimately a scalar. However, ``BraKet.create`` may return scalars
        directly, which can be used in place of operators in most expression,
        owing to the :func:`~qnet.algebra.operator_algebra.scalars_to_op`
        simplification rule.
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


class KetIndexedSum(IndexedSum, Ket):
    # Order of superclasses is important for proper mro for __add__ etc.
    # (we're using cooperative inheritance from both superclasses,
    # cf. https://stackoverflow.com/q/47804919)

    # TODO: documentation

    _rules = OrderedDict()  # see end of module
    _simplifications = [
        assoc_indexed, indexed_sum_over_const, indexed_sum_over_kronecker,
        match_replace, ]

    @property
    def space(self):
        return self.term.space

    def _expand(self):
        return self.__class__.create(self.term.expand(), *self.ranges)

    def _series_expand(self, param, about, order):
        raise NotImplementedError()

    def _adjoint(self):
        return self.__class__.create(self.term.adjoint(), *self.ranges)

    def __mul__(self, other):
        if isinstance(other, Bra):
            if isinstance(other.ket, KetIndexedSum):
                other_ket = other.ket.make_disjunct_indices(self)
                assert isinstance(other_ket, KetIndexedSum)
                new_ranges = self.ranges + other_ket.ranges
                new_term = KetBra.create(self.term, other_ket.term)
                return OperatorIndexedSum(new_term, *new_ranges)
            else:
                return OperatorIndexedSum(
                    KetBra.create(self.term, other.ket), *self.ranges)
        elif isinstance(other, Ket):
            if not isinstance(other, KetIndexedSum):
                return KetIndexedSum(self.term * other, *self.ranges)
            # isinstance(other, KetIndexedSum) is handled by IndexedSum.__mul__
        try:
            return super().__mul__(other)
        except AttributeError:
            return NotImplemented
