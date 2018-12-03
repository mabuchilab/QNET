r"""This module implements the algebra of states in a Hilbert space

For more details see :ref:`state_algebra`.
"""
import re
from abc import ABCMeta
from collections import OrderedDict
from itertools import product as cartesian_product

import sympy

from .abstract_algebra import Expression, Operation
from .scalar_algebra import is_scalar
from .abstract_quantum_algebra import (
    ScalarTimesQuantumExpression, QuantumExpression, QuantumSymbol,
    QuantumPlus, QuantumTimes, QuantumAdjoint, QuantumIndexedSum,
    QuantumDerivative, ensure_local_space, _series_expand_combine_prod)
from .algebraic_properties import (
    accept_bras, assoc, assoc_indexed, basis_ket_zero_outside_hs,
    filter_neutral, match_replace, match_replace_binary, orderby,
    collect_summands)
from .exceptions import OverlappingSpaces, UnequalSpaces, SpaceTooLargeError
from .hilbert_space_algebra import FullSpace, TrivialSpace
from qnet.algebra.core.algebraic_properties import (
    indexed_sum_over_const,
    indexed_sum_over_kronecker)
from .operator_algebra import Operator, OperatorPlus, PseudoInverse
from .scalar_algebra import ScalarExpression
from ...utils.indices import (
    FockIndex, IdxSym, IndexOverFockSpace, IndexOverRange, SymbolicLabelBase, )
from ...utils.ordering import FullCommutativeHSOrder
from ...utils.singleton import Singleton, singleton_object

__all__ = [
    'BasisKet', 'Bra', 'BraKet', 'CoherentStateKet', 'State', 'KetBra',
    'KetPlus', 'KetSymbol', 'LocalKet', 'OperatorTimesKet', 'ScalarTimesKet',
    'TensorKet', 'TrivialKet', 'ZeroKet', 'KetIndexedSum', 'StateDerivative']

__private__ = []  # anything not in __all__ must be in __private__


###############################################################################
# Algebraic properties
###############################################################################


class State(QuantumExpression, metaclass=ABCMeta):
    """Base class for states in a Hilbert space"""

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
        # In the State algebra, any adjoint is a Bra
        # instance at the top level. To enforce this, we need some custom
        # __mul__ implementations, in particular for KetIndexedSums

        def check_if_sum(*args):
            return [isinstance(arg, KetIndexedSum) for arg in args]

        if isinstance(other, State):
            isket = (self.isket, other.isket)
            if isket == (True, True):
                # Ket * Ket
                is_sum = check_if_sum(self, other)
                if is_sum == [True, True]:
                    return QuantumIndexedSum.__mul__(self, other)
                elif is_sum == [True, False]:
                    return QuantumIndexedSum.__mul__(self, other)
                elif is_sum == [False, True]:
                    return QuantumIndexedSum.__rmul__(other, self)
                elif is_sum == [False, False]:
                    return TensorKet.create(self, other)
            elif isket == (True, False):
                # Ket * Bra
                is_sum = check_if_sum(self, other.ket)
                if is_sum == [True, True]:
                    other_ket = other.ket.make_disjunct_indices(self)
                    new_term = self.term * Bra(other_ket.term)
                    new_ranges = self.ranges + other_ket.ranges
                    return new_term.__class__._indexed_sum_cls.create(
                        new_term, *new_ranges)
                elif is_sum == [True, False]:
                    return QuantumIndexedSum.__mul__(self, other)
                elif is_sum == [False, True]:
                    new_term = self * Bra(other.ket.term)
                    return new_term.__class__._indexed_sum_cls.create(
                        new_term, *other.ket.ranges)
                elif is_sum == [False, False]:
                    return KetBra.create(self, other.ket)
            elif isket == (False, True):
                # Bra * Ket
                is_sum = check_if_sum(self.ket, other)
                if is_sum == [True, True]:
                    other = other.make_disjunct_indices(self.ket)
                    new_term = Bra(self.ket.term) * other.term
                    new_ranges = self.ket.ranges + other.ranges
                    return new_term.__class__._indexed_sum_cls.create(
                        new_term, *new_ranges)
                elif is_sum == [True, False]:
                    new_term = Bra(self.ket.term) * other
                    return new_term.__class__._indexed_sum_cls.create(
                        new_term, *self.ket.ranges)
                elif is_sum == [False, True]:
                    return QuantumIndexedSum.__rmul__(other, self)
                elif is_sum == [False, False]:
                    return BraKet.create(self.ket, other)
            elif isket == (False, False):
                # Bra * Bra
                return Bra.create(self.ket * other.ket)
        elif isinstance(other, Operator):
            if self.isbra:
                return Bra.create(other.adjoint() * self.ket)
        try:
            return super().__mul__(other)
        except AttributeError:
            return NotImplemented

    def __rmul__(self, other):
        if self.isket and isinstance(other, Operator):
            return OperatorTimesKet.create(other, self)
        elif self.isbra and is_scalar(other):
            return Bra(other.conjugate() * self.ket)
        try:
            return super().__rmul__(other)
        except AttributeError:
            return NotImplemented


###############################################################################
# Operator algebra elements
###############################################################################


class KetSymbol(QuantumSymbol, State):
    """Symbolic state

    See :class:`.QuantumSymbol`.
    """
    _rx_label = re.compile(
        r'(^[+-]?\d+(/\d+)?$|'
        r'^[A-Za-z0-9+-]+([A-Za-z0-9()_,.+-=]+)?$)')


class LocalKet(State, metaclass=ABCMeta):
    """A state on a :class:`LocalSpace`

    This does not include operations, even if these operations only involve
    states acting on the same local space"""

    def __init__(self, *args, hs):
        hs = ensure_local_space(hs, cls=self._default_hs_cls)
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

    def _diff(self, sym):
        return self


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

    def _diff(self, sym):
        return ZeroKet


class BasisKet(LocalKet, KetSymbol):
    """Local basis state, identified by index or label

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

    simplifications = [basis_ket_zero_outside_hs]

    def __init__(self, label_or_index, *, hs):
        hs = ensure_local_space(hs, cls=self._default_hs_cls)
        label, ind = hs._unpack_basis_label_or_index(label_or_index)
        self._index = ind
        super().__init__(label, hs=hs)

    @property
    def args(self):
        """Tuple containing `label_or_index` as its only element."""
        if self.space.has_basis or isinstance(self.label, SymbolicLabelBase):
            return (self.label, )
        else:
            return (self.index, )

    @property
    def index(self):
        """The index of the state in the Hilbert space basis

        >>> hs = LocalSpace('tls', basis=('g', 'e'))
        >>> BasisKet('g', hs=hs).index
        0
        >>> BasisKet('e', hs=hs).index
        1
        >>> BasisKet(1, hs=hs).index
        1

        For a :class:`BasisKet` with an indexed label, this may return a sympy
        expression::

        >>> hs = SpinSpace('s', spin='3/2')
        >>> i = symbols('i', cls=IdxSym)
        >>> lbl = SpinIndex(i/2, hs)
        >>> ket = BasisKet(lbl, hs=hs)
        >>> ket.index
        i/2 + 3/2
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
        if isinstance(self.label, SymbolicLabelBase):
            next_label = self.space.next_basis_label_or_index(
                self.label, n)
            return BasisKet(next_label, hs=self.space)
        else:
            try:
                next_index = self.space.next_basis_label_or_index(
                    self.index, n)
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
    """Local coherent state, labeled by a complex amplitude

    Args:
        hs (LocalSpace): The local Hilbert space degree of freedom.
        ampl (Scalar): The coherent displacement amplitude.
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

    def _diff(self, sym):
        from qnet.algebra.library.fock_operators import Destroy, Create
        hs = self.space
        return (
            (self._ampl * Create(hs=hs) -
             self._ampl.conjugate() * Destroy(hs=hs))
            .diff(sym)
            * self)

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
    """Sum of states"""
    _neutral_element = ZeroKet
    _binary_rules = OrderedDict()
    simplifications = [
        accept_bras, assoc, orderby, collect_summands, match_replace_binary]

    order_key = FullCommutativeHSOrder

    def __init__(self, *operands):
        _check_kets(*operands, same_space=True)
        super().__init__(*operands)


class TensorKet(State, QuantumTimes):
    """A tensor product of kets

    Each ket must belong to different degree of freedom (:class:`.LocalSpace`).
    """
    _binary_rules = OrderedDict()
    _neutral_element = TrivialKet
    simplifications = [
        accept_bras, assoc, orderby, filter_neutral, match_replace_binary]

    order_key = FullCommutativeHSOrder

    def __init__(self, *operands):
        _check_kets(*operands, disjunct_space=True)
        super().__init__(*operands)

    @classmethod
    def create(cls, *ops):
        if any(o == ZeroKet for o in ops):
            return ZeroKet
        return super().create(*ops)


class ScalarTimesKet(State, ScalarTimesQuantumExpression):
    """Product of a :class:`.Scalar` coefficient and a ket

    Args:
        coeff (Scalar): coefficient
        term (State): the ket that is multiplied
    """
    _rules = OrderedDict()
    simplifications = [match_replace, ]

    @classmethod
    def create(cls, coeff, term):
        if term.isbra:
            scalar_times_ket = coeff.conjugate() * term.ket
            return Bra.create(scalar_times_ket)
        return super().create(coeff, term)

    def __init__(self, coeff, term):
        _check_kets(term)
        super().__init__(coeff, term)


class OperatorTimesKet(State, Operation):
    """Product of an operator and a state."""
    _rules = OrderedDict()
    simplifications = [match_replace]

    def __init__(self, operator, ket):
        _check_kets(ket)
        if not operator.space <= ket.space:
            raise SpaceTooLargeError(
                str(operator.space) + " <!= " + str(ket.space))
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

    def _diff(self, sym):
        return (
            self.operator.diff(sym) * self.ket +
            self.operator * self.ket.diff(sym))

    def _series_expand(self, param, about, order):
        ce = self.operator.series_expand(param, about, order)
        te = self.ket.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class StateDerivative(QuantumDerivative, State):
    """Symbolic partial derivative of a state

    See :class:`.QuantumDerivative`.
    """
    pass


class Bra(State, QuantumAdjoint):
    """The associated dual/adjoint state for any ket"""
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
        if is_scalar(other):
            return Bra.create(self.ket/other.conjugate())
        return NotImplemented


class BraKet(ScalarExpression, Operation):
    r"""The symbolic inner product between two states

    This mathermatically corresponds to:

    .. math::
        \langle b | k \rangle

    which we define to be linear in the state :math:`k` and anti-linear in
    :math:`b`.

    Args:
        bra (State): The anti-linear state argument. Note that this is *not* a
            :class:`Bra` instance.
        ket (State): The linear state argument.
    """
    _rules = OrderedDict()
    _space = TrivialSpace
    simplifications = [match_replace]

    def __init__(self, bra, ket):
        _check_kets(bra, ket, same_space=True)
        super().__init__(bra, ket)

    @property
    def ket(self):
        """The ket of the braket"""
        return self.operands[1]

    @property
    def bra(self):
        """The bra of the braket (:class:`Bra` instance)"""
        return Bra(self.operands[0])

    def _diff(self, sym):
        bra, ket = self.operands
        return (
            self.__class__.create(bra.diff(sym), ket) +
            self.__class__.create(bra, ket.diff(sym)))

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
        return _series_expand_combine_prod(be, ke, order)


class KetBra(Operator, Operation):
    """Outer product of two states

    Args:
        ket (State): The left factor in the product
        bra (State): The right factor in the product. Note that this is *not* a
            :class:`Bra` instance.
    """
    _rules = OrderedDict()
    simplifications = [match_replace]

    def __init__(self, ket, bra):
        _check_kets(ket, bra, same_space=True)
        super().__init__(ket, bra)

    @property
    def ket(self):
        """The left factor in the product"""
        return self.operands[0]

    @property
    def bra(self):
        """The co-state right factor in the product

        This is a :class:`Bra` instance (unlike the `bra` given to the
        constructor
        """
        return Bra(self.operands[1])

    @property
    def space(self):
        """The Hilbert space of the states being multiplied"""
        return self.operands[0].space

    def _adjoint(self):
        return KetBra.create(*reversed(self.operands))

    def _pseudo_inverse(self):
        return PseudoInverse(self)

    def _diff(self, sym):
        ket, bra = self.operands
        return (
            self.__class__.create(ket.diff(sym), bra) +
            self.__class__.create(ket, bra.diff(sym)))

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


class KetIndexedSum(State, QuantumIndexedSum):
    """Indexed sum over Kets"""
    # Must inherit from State first, so that proper __mul__ is used

    _rules = OrderedDict()
    simplifications = [
        assoc_indexed, indexed_sum_over_kronecker, indexed_sum_over_const,
        match_replace, ]

    @classmethod
    def create(cls, term, *ranges):
        if term.isbra:
            return Bra.create(KetIndexedSum.create(term.ket, *ranges))
        else:
            return super().create(term, *ranges)

    def __init__(self, term, *ranges):
        _check_kets(term)
        super().__init__(term, *ranges)


def _check_kets(*ops, same_space=False, disjunct_space=False):
    """Check that all operands are Kets from the same Hilbert space."""
    if not all([(isinstance(o, State) and o.isket) for o in ops]):
        raise TypeError("All operands must be Kets")
    if same_space:
        if not len({o.space for o in ops if o is not ZeroKet}) == 1:
            raise UnequalSpaces(str(ops))
    if disjunct_space:
        spc = TrivialSpace
        for o in ops:
            if o.space & spc > TrivialSpace:
                raise OverlappingSpaces(str(ops))
            spc *= o.space


State._zero = ZeroKet
State._one = TrivialKet
State._base_cls = State
State._scalar_times_expr_cls = ScalarTimesKet
State._plus_cls = KetPlus
State._times_cls = TensorKet
State._adjoint_cls = Bra
State._indexed_sum_cls = KetIndexedSum
State._derivative_cls = StateDerivative
