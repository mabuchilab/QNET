r"""This module implements the algebra of states in a Hilbert space

For more details see :ref:`state_algebra`.
"""
import re
from abc import ABCMeta
from collections import OrderedDict
from itertools import product as cartesian_product

import sympy

from .abstract_algebra import Expression, Operation
from .abstract_quantum_algebra import (
    ScalarTimesQuantumExpression, QuantumExpression, QuantumSymbol,
    QuantumPlus, QuantumTimes, QuantumAdjoint, QuantumIndexedSum,
    ensure_local_space)
from .algebraic_properties import (
    accept_bras, assoc, assoc_indexed, basis_ket_zero_outside_hs,
    check_kets_same_space, check_op_ket_space, filter_neutral, match_replace,
    match_replace_binary, orderby, )
from .exceptions import OverlappingSpaces, UnequalSpaces
from .hilbert_space_algebra import FullSpace, TrivialSpace
from qnet.algebra.core.algebraic_properties import (
    indexed_sum_over_const,
    indexed_sum_over_kronecker)
from .operator_algebra import Operator, OperatorPlus
from .scalar_types import SCALAR_TYPES
from ...utils.indices import (
    FockIndex, IdxSym, IndexOverFockSpace, IndexOverRange, SymbolicLabelBase, )
from ...utils.ordering import FullCommutativeHSOrder
from ...utils.singleton import Singleton, singleton_object

__all__ = [
    'BasisKet', 'Bra', 'BraKet', 'CoherentStateKet', 'State', 'KetBra',
    'KetPlus', 'KetSymbol', 'LocalKet', 'OperatorTimesKet', 'ScalarTimesKet',
    'TensorKet', 'TrivialKet', 'ZeroKet', 'KetIndexedSum']

__private__ = []  # anything not in __all__ must be in __private__


###############################################################################
# Algebraic properties
###############################################################################


class State(QuantumExpression, metaclass=ABCMeta):
    """Basic State algebra class to represent Hilbert Space states"""

    def _adjoint(self):
        if self.isket:
            return Bra(self)
        else:
            return self.ket

    @property
    def isket(self):
        """Whether the state represents a ket"""
        # We enforce that all sub-classes of Ket except Bra are normal
        # Hilbert space vectors. For example, KetPlus(Bra(...), Bra(...))
        # is disallowed, and is represented as Bra(KetPlus(.., ...)). Thus,
        # all State instances that are not directly an instance of Bra are
        # kets. Name of property 'isket' is taken from QuTiP
        return not isinstance(self, Bra)

    @property
    def isbra(self):
        """Wether the state represents a bra (adjoint ket)"""
        return isinstance(self, Bra)

    @property
    def bra(self):
        """The bra associated with a ket"""
        if self.isbra:
            return self
        else:
            return self._adjoint()

    @property
    def ket(self):
        """The ket associated with a bra"""
        if self.isket:
            return self
        else:
            return self._adjoint()

    def __mul__(self, other):
        if isinstance(other, State):
            if isinstance(other, KetIndexedSum):
                return other.__rmul__(self)
            isket = (self.isket, other.isket)
            if isket == (True, False):
                return KetBra.create(self, other.ket)
            elif isket == (False, True):
                return BraKet.create(self.ket, other)
            elif isket == (False, False):
                return Bra.create(self.ket * other.ket)
        elif isinstance(other, Operator):
            if self.isbra:
                return Bra.create(other * self.ket)
        try:
            return super().__mul__(other)
        except AttributeError:
            return NotImplemented

    def __rmul__(self, other):
        if self.isket and isinstance(other, Operator):
            return OperatorTimesKet.create(other, self)
        elif self.isbra and isinstance(other, SCALAR_TYPES):
            return Bra(other.conjugate() * self.ket)
        try:
            return super().__rmul__(other)
        except AttributeError:
            return NotImplemented


###############################################################################
# Operator algebra elements
###############################################################################


class KetSymbol(QuantumSymbol, State):
    """Symbolic state"""
    _rx_label = re.compile('^[A-Za-z0-9]+(_[A-Za-z0-9().+-]+)?$')


class LocalKet(State, metaclass=ABCMeta):
    """A state that lives on a single local Hilbert space. This does
    not include operations, even if these operations only involve states acting
    on the same local space"""

    def __init__(self, *args, hs):
        hs = ensure_local_space(hs)
        self._hs = hs
        super().__init__(*args, hs=hs)

    @property
    def space(self):
        return self._hs

    @property
    def kwargs(self):
        return {'hs': self._hs}


@singleton_object
class ZeroKet(State, Expression, metaclass=Singleton):
    """ZeroKet constant (singleton) object for the null-state."""

    _order_index = 2

    @property
    def space(self):
        return FullSpace

    @property
    def args(self):
        return tuple()

    def __eq__(self, other):
        return self is other or other == 0

    def all_symbols(self):
        return set([])


@singleton_object
class TrivialKet(State, Expression, metaclass=Singleton):
    """TrivialKet constant (singleton) object.
    This is the neutral element under the state tensor-product.
    """

    _order_index = 2

    @property
    def space(self):
        return TrivialSpace

    def _adjoint(self):
        return Bra(TrivialKet)

    @property
    def args(self):
        return tuple()

    def __eq__(self, other):
        return self is other or other == 1

    def all_symbols(self):
        return set([])


class BasisKet(LocalKet, KetSymbol):
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
        hs = ensure_local_space(hs)
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
    :param SCALAR_TYPES ampl: The coherent displacement amplitude.
    """

    _rx_label = re.compile('^.*$')

    def __init__(self, ampl, *, hs):
        self._ampl = ampl
        super().__init__(ampl, hs=hs)

    @property
    def args(self):
        return (self._ampl, )


    @property
    def ampl(self):
        return self._ampl

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


class KetPlus(State, QuantumPlus):
    """A sum of states."""
    neutral_element = ZeroKet
    _binary_rules = OrderedDict()
    _simplifications = [
        accept_bras, assoc, orderby, filter_neutral, match_replace_binary]

    order_key = FullCommutativeHSOrder

    def __init__(self, *operands):
        if not all([o.isket for o in operands]):
            raise TypeError("All operands must be Kets")
        if not len({o.space for o in operands if o is not ZeroKet}) == 1:
            raise UnequalSpaces(str(operands))
        super().__init__(*operands)


class TensorKet(State, QuantumTimes):
    """A tensor product of kets each belonging to different degrees of freedom.
    """
    _binary_rules = OrderedDict()
    neutral_element = TrivialKet
    _simplifications = [
        accept_bras, assoc, orderby, filter_neutral, match_replace_binary]

    order_key = FullCommutativeHSOrder

    def __init__(self, *operands):
        spc = TrivialSpace
        for o in operands:
            if o.space & spc > TrivialSpace:
                raise OverlappingSpaces(str(operands))
            spc *= o.space
        super().__init__(*operands)

    @classmethod
    def create(cls, *ops):
        if any(o == ZeroKet for o in ops):
            return ZeroKet
        return super().create(*ops)


class ScalarTimesKet(State, ScalarTimesQuantumExpression):
    """Multiply a Ket by a scalar coefficient.

    Args:
        coeff (SCALAR_TYPES): coefficient
        term (State): the ket that is multiplied
    """
    _rules = OrderedDict()
    _simplifications = [match_replace, ]

    def __init__(self, coeff, term):
        if not term.isket:
            raise TypeError("term must be a ket")
        super().__init__(coeff, term)


class OperatorTimesKet(State, Operation):
    """Product of an operator and a state."""
    _rules = OrderedDict()
    _simplifications = [match_replace, check_op_ket_space]

    def __init__(self, operator, ket):
        if ket.isbra:
            raise TypeError("ket cannot be a Bra instance")
        super().__init__(operator, ket)

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
        ce = self.operator.series_expand(param, about, order)
        te = self.ket.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class Bra(State, QuantumAdjoint):
    """The associated dual/adjoint state for any `ket`"""
    def __init__(self, ket):
        if ket.isbra:
            raise TypeError("ket cannot be a Bra instance")
        super().__init__(ket)

    @property
    def ket(self):
        """The original :class:`State`"""
        return self.operands[0]

    @property
    def bra(self):
        return self

    operand = ket

    @property
    def isket(self):
        """False, by defintion"""
        return False

    @property
    def isbra(self):
        """True, by definition"""
        return True

    def _adjoint(self):
        return self.ket

    @property
    def label(self):
        return self.ket.label

    def __add__(self, other):
        if isinstance(other, Bra):
            return Bra.create(self.ket + other.ket)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Bra):
            return Bra.create(self.ket - other.ket)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return Bra.create(self.ket/other.conjugate())
        return NotImplemented


class BraKet(Operator, Operation):
    r"""The symbolic inner product between two states, represented as Bra and
    Ket

    In math notation this corresponds to:

    .. math::
        \langle b | k \rangle

    which we define to be linear in the state :math:`k` and anti-linear in
    :math:`b`.

    Args:
        bra (State): The anti-linear state argument. Note that this is *not* a
            :class:`Bra` instance.
        ket (State): The linear state argument.

    Note:
        :class:`Braket` is an :class:`Operator` in the
        :class:`~qnet.algebra.hilbert_space_algebra.TrivialSpace`, that is,
        ultimately a scalar. However, ``BraKet.create`` may return scalars
        directly, which can be used in place of operators in most expression,
        owing to the :func:`~qnet.algebra.operator_algebra.scalars_to_op`
        simplification rule.
    """
    _rules = OrderedDict()
    _space = TrivialSpace
    _simplifications = [check_kets_same_space, match_replace]

    def __init__(self, bra, ket):
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
    """A symbolic operator formed by the outer product of two states"""
    _rules = OrderedDict()
    _simplifications = [check_kets_same_space, match_replace]

    def __init__(self, ket, bra):
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


class KetIndexedSum(QuantumIndexedSum, State):
    """Indexed sum over Kets"""

    _rules = OrderedDict()
    _simplifications = [
        assoc_indexed, indexed_sum_over_const, indexed_sum_over_kronecker,
        match_replace, ]


State._zero = ZeroKet
State._one = TrivialKet
State._base_cls = State
State._scalar_times_expr_cls = ScalarTimesKet
State._plus_cls = KetPlus
State._times_cls = TensorKet
State._adjoint_cls = Bra
State._indexed_sum_cls = KetIndexedSum
