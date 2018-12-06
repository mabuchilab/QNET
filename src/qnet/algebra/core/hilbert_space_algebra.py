r"""Core class hierarchy for Hilbert spaces

This module defines some simple classes to describe simple and
*composite/tensor* (i.e., multiple degree of freedom)
Hilbert spaces of quantum systems.

For more details see :ref:`hilbert_space_algebra`.
"""
import functools
import operator
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product as cartesian_product

from .abstract_algebra import (
    Expression, Operation, )
from .algebraic_properties import (
    assoc, idem, filter_neutral, convert_to_spaces, empty_trivial, )
from .exceptions import AlgebraError, BasisNotSetError
from ...utils.indices import SymbolicLabelBase, FockIndex, FockLabel, StrLabel
from ...utils.ordering import KeyTuple
from ...utils.singleton import Singleton, singleton_object

__all__ = [
    'HilbertSpace', 'LocalSpace', 'ProductSpace',
    'FullSpace', 'TrivialSpace']

__private__ = []  # anything not in __all__ must be in __private__


###############################################################################
# Abstract base classes
###############################################################################


class HilbertSpace(metaclass=ABCMeta):
    """Base class for Hilbert spaces"""

    def tensor(self, *others):
        """Tensor product between Hilbert spaces"""
        return ProductSpace.create(self, *others)

    @abstractmethod
    def remove(self, other):
        """Remove a particular factor from a tensor product space."""
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def intersect(self, other):
        """Find the mutual tensor factors of two Hilbert spaces."""
        raise NotImplementedError(self.__class__.__name__)

    @property
    @abstractmethod
    def local_factors(self):
        """Return tuple of LocalSpace objects that tensored together yield this
        Hilbert space."""
        raise NotImplementedError(self.__class__.__name__)

    def isdisjoint(self, other):
        """Check whether two Hilbert spaces are disjoint (do not have any
        common local factors). Note that `FullSpace` is *not* disjoint with any
        other Hilbert space, while `TrivialSpace` *is* disjoint with any other
        HilbertSpace (even itself)
        """
        if other == FullSpace:
            return False
        else:
            for ls in self.local_factors:
                if isinstance(ls.label, StrLabel):
                    return False
            for ls in other.local_factors:
                if isinstance(ls.label, StrLabel):
                    return False
            return set(self.local_factors).isdisjoint(set(other.local_factors))

    def is_tensor_factor_of(self, other):
        """Test if a space is included within a larger tensor product space.
        Also ``True`` if ``self == other``.

        :param other: Other Hilbert space
        :type other: HilbertSpace
        :rtype: bool
        """
        return self <= other

    def is_strict_tensor_factor_of(self, other):
        """Test if a space is included within a larger tensor product space.
        Not ``True`` if ``self == other``."""
        return self < other

    @property
    def dimension(self):
        """Full dimension of the Hilbert space.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    @property
    def has_basis(self):
        """True if the Hilbert space has a basis"""
        return False

    @property
    def basis_states(self):
        """Yield an iterator over the states (:class:`.State` instances) that
        form the canonical basis of the Hilbert space

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    def basis_state(self, index_or_label):
        """Return the basis state with the given index or label.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
            IndexError: if there is no basis state with the given index
            KeyError: if there is not basis state with the given label
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    @property
    def basis_labels(self):
        """Tuple of basis labels.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    @abstractmethod
    def is_strict_subfactor_of(self, other):
        """Test whether a Hilbert space occures as a strict sub-factor in
        a (larger) Hilbert space"""
        raise NotImplementedError(self.__class__.__name__)

    def __len__(self):
        """The number of LocalSpace factors / degrees of freedom."""
        return len(self.local_factors)

    def __mul__(self, other):
        return self.tensor(other)

    def __truediv__(self, other):
        return self.remove(other)

    def __and__(self, other):
        return self.intersect(other)

    def __lt__(self, other):
        return self.is_strict_subfactor_of(other)

    def __gt__(self, other):
        return other.is_strict_subfactor_of(self)

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return self == other or self > other


###############################################################################
# Hilbert space algebra elements
###############################################################################


class LocalSpace(HilbertSpace, Expression):
    """Hilbert space for a single degree of freedom.

    Args:
        label (str or int or StrLabel): label (subscript) of the
            Hilbert space
        basis (tuple or None): Set an explicit basis for the Hilbert space
            (tuple of labels for the basis states)
        dimension (int or None): Specify the dimension $n$ of the Hilbert
            space.  This implies a basis numbered from 0  to $n-1$.
        local_identifiers (dict): Mapping of class names of
            :class:`.LocalOperator` subclasses to identifier names. Used e.g.
            'b' instead of the default 'a' for the anihilation operator. This
            can be a dict or a dict-compatible structure, e.g. a list/tuple of
            key-value tuples.
        order_index (int or None): An optional key that determines the
            preferred order of Hilbert spaces. This also changes the order of
            e.g. sums or products of Operators. Hilbert spaces will be ordered
            from left to right be increasing `order_index`; Hilbert spaces
            without an explicit `order_index` are sorted by their label

    A :class:`LocalSpace` fundamentally has a Fock-space like structure,
    in that its basis states may be understood as an "excitation".
    The spectrum can be infinite, with levels labeled by integers 0, 1, ...::

        >>> hs = LocalSpace(label=0)

    or truncated to a finite dimension::

        >>> hs = LocalSpace(0, dimension=5)
        >>> hs.basis_labels
        ('0', '1', '2', '3', '4')

    For finite-dimensional (truncated) Hilbert spaces, we also allow an
    arbitrary alternative labeling of the canonical basis::

        >>> hs = LocalSpace('rydberg', dimension=3, basis=('g', 'e', 'r'))

    """
    _rx_label = re.compile('^[A-Za-z0-9.+-]+(_[A-Za-z0-9().+-]+)?$')

    _basis_label_types = (int, str, FockIndex, FockLabel)
    # acceptable types for labels

    def __init__(
            self, label, *, basis=None, dimension=None, local_identifiers=None,
            order_index=None):

        default_args = []
        if basis is None:
            default_args.append('basis')
        else:
            basis = tuple([str(label) for label in basis])
        if dimension is None:
            default_args.append('dimension')
        else:
            dimension = int(dimension)
        if order_index in [None, float('inf')]:
            default_args.append('order_index')
            order_index = float('inf')  # ensure sort as last
        else:
            order_index = int(order_index)
        if local_identifiers is None:
            default_args.append('local_identifiers')
            local_identifiers = {}
        else:
            local_identifiers = dict(local_identifiers)
        try:
            # we want to normalize the local_identifiers to an arbitrary stable
            # order
            self._sorted_local_identifiers = tuple(
                sorted(tuple(local_identifiers.items())))
        except TypeError:
            # this will happen e.g. if the keys in local_identifier are types
            # instead of class names
            raise TypeError(
                "local_identifier must map class names to identifier strings")

        if not isinstance(label, StrLabel):
            label = str(label)
            if not self._rx_label.match(label):
                raise ValueError(
                    "label '%s' does not match pattern '%s'"
                    % (label, self._rx_label.pattern))
        if basis is None:
            if dimension is not None:
                basis = tuple([str(i) for i in range(dimension)])
        else:
            if dimension is None:
                dimension = len(basis)
            else:
                if dimension != len(basis):
                    raise ValueError("basis and dimension are incompatible")

        self._label = label
        self._order_key = KeyTuple((
            order_index, label, str(dimension), basis,
            self._sorted_local_identifiers))
        self._basis = basis
        self._dimension = dimension
        self._local_identifiers = local_identifiers
        self._order_index = order_index
        self._kwargs = OrderedDict([
            ('basis', self._basis), ('dimension', self._dimension),
            ('local_identifiers', self._sorted_local_identifiers),
            ('order_index', self._order_index)])
        self._minimal_kwargs = self._kwargs.copy()
        for key in default_args:
            del self._minimal_kwargs[key]

        super().__init__(
            label, basis=basis, dimension=dimension,
            local_identifiers=self._sorted_local_identifiers,
            order_index=order_index)

    @classmethod
    def _check_basis_label_type(cls, label_or_index):
        """Every object (BasisKet, LocalSigma) that contains a label or index
        for an eigenstate of some LocalSpace should call this routine to check
        the type of that label or index (or, use
        :meth:`_unpack_basis_label_or_index`"""
        if not isinstance(label_or_index, cls._basis_label_types):
            raise TypeError(
                "label_or_index must be an instance of one of %s; not %s" % (
                    ", ".join([t.__name__ for t in cls._basis_label_types]),
                    label_or_index.__class__.__name__))

    def _unpack_basis_label_or_index(self, label_or_index):
        """return tuple (label, ind) from `label_or_index`

        If `label_or_int` is a :class:`.SymbolicLabelBase` sub-instance, it
        will be stored in the `label` attribute, and the `ind` attribute will
        return the value of the label's :attr:`.FockIndex.fock_index`
        attribute.  No checks are performed for symbolic labels.

        :meth:`_check_basis_label_type` is called on `label_or_index`.

        Raises:
            ValueError: if `label_or_index` is a :class:`str` referencing an
                invalid basis state; or, if `label_or_index` is an :class:`int`
                < 0 or >= the dimension of the Hilbert space
            BasisNotSetError: if `label_or_index` is a :class:`str`, but the
                Hilbert space has no defined basis
            TypeError: if `label_or_int` is not a :class:`str`, :class:`int`,
                or :class:`.SymbolicLabelBase`, or more generally whatever
                types are allowed through the `_basis_label_types` attribute of
                the Hilbert space.
        """
        self._check_basis_label_type(label_or_index)
        if isinstance(label_or_index, str):
            label = label_or_index
            try:
                ind = self.basis_labels.index(label)
                # the above line may also raise BasisNotSetError, which we
                # don't want to catch here
            except ValueError:
                # a less confusing error message:
                raise ValueError(
                    "%r is not one of the basis labels %r"
                    % (label, self.basis_labels))
        elif isinstance(label_or_index, int):
            ind = label_or_index
            if ind < 0:
                raise ValueError("Index %d must be >= 0" % ind)
            if self.has_basis:
                if ind >= self.dimension:
                    raise ValueError(
                        "Index %s must be < the dimension %d of Hilbert "
                        "space %s" % (ind, self.dimension, self))
                label = self.basis_labels[label_or_index]
            else:
                label = str(label_or_index)
        elif isinstance(label_or_index, SymbolicLabelBase):
            label = label_or_index
            try:
                ind = label_or_index.fock_index
            except AttributeError:
                raise TypeError(
                    "label_or_index must define a fock_index attribute in "
                    "order to be used for identifying a level in a Hilbert "
                    "space")
        else:
            raise TypeError(
                "label_or_index must be an int or str, or SymbolicLabelBase, "
                "not %s" % type(label_or_index))
        return label, ind

    @property
    def args(self):
        """List of arguments, consisting only of `label`"""
        return (self._label, )

    @property
    def label(self):
        """Label of the Hilbert space"""
        return self._label

    @property
    def has_basis(self):
        """True if the Hilbert space has a basis"""
        return self._basis is not None

    @property
    def basis_states(self):
        """Yield an iterator over the states (:class:`.BasisKet` instances)
        that form the canonical basis of the Hilbert space

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        from qnet.algebra.core.state_algebra import BasisKet  # avoid circ. import
        for label in self.basis_labels:
            yield BasisKet(label, hs=self)

    def basis_state(self, index_or_label):
        """Return the basis state with the given index or label.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
            IndexError: if there is no basis state with the given index
            KeyError: if there is not basis state with the given label
        """
        from qnet.algebra.core.state_algebra import BasisKet  # avoid circ. import
        try:
            return BasisKet(index_or_label, hs=self)
        except ValueError as exc_info:
            if isinstance(index_or_label, int):
                raise IndexError(str(exc_info))
            else:
                raise KeyError(str(exc_info))

    @property
    def basis_labels(self):
        """Tuple of basis labels (strings).

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        if self._basis is None:
            raise BasisNotSetError(
                "Hilbert space %s has no defined basis" % str(self))
        return self._basis

    @property
    def dimension(self):
        """Dimension of the Hilbert space.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        if self._dimension is not None:
            return self._dimension
        else:
            raise BasisNotSetError(
                "Hilbert space %s has no defined basis" % str(self))

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def minimal_kwargs(self):
        return self._minimal_kwargs

    def remove(self, other):
        if other == self:
            return TrivialSpace
        return self

    def intersect(self, other):
        if self in other.local_factors:
            return self
        return TrivialSpace

    @property
    def local_factors(self):
        return (self, )

    def is_strict_subfactor_of(self, other):
        if isinstance(other, ProductSpace) and self in other.operands:
            assert len(other.operands) > 1
            return True
        if other is FullSpace:
            return True
        return False

    def next_basis_label_or_index(self, label_or_index, n=1):
        """Given the label or index of a basis state, return the label/index of
        the next basis state.

        More generally, if `n` is given, return the `n`'th next basis state
        label/index; `n` may also be negative to obtain previous basis state
        labels/indices.

        The return type is the same as the type of `label_or_index`.

        Args:
            label_or_index (int or str or SymbolicLabelBase): If `int`, the
                index of a basis state; if `str`, the label of a basis state
            n (int): The increment

        Raises:
            IndexError: If going beyond the last or first basis state
            ValueError: If `label` is not a label for any basis state in the
                Hilbert space
            .BasisNotSetError: If the Hilbert space has no defined basis
            TypeError: if `label_or_index` is neither a :class:`str` nor an
                :class:`int`, nor a :class:`SymbolicLabelBase`
        """
        if isinstance(label_or_index, int):
            new_index = label_or_index + n
            if new_index < 0:
                raise IndexError("index %d < 0" % new_index)
            if self.has_basis:
                if new_index >= self.dimension:
                    raise IndexError("index %d out of range for basis %s"
                                     % (new_index, self._basis))
            return new_index
        elif isinstance(label_or_index, str):
            label_index = self.basis_labels.index(label_or_index)
            new_index = label_index + n
            if (new_index < 0) or (new_index >= len(self._basis)):
                raise IndexError("index %d out of range for basis %s"
                                 % (new_index, self._basis))
            return self._basis[new_index]
        elif isinstance(label_or_index, SymbolicLabelBase):
            return label_or_index.__class__(expr=label_or_index.expr + n)
        else:
            raise TypeError(
                "Invalid type for label_or_index: %s"
                % label_or_index.__class__.__name__)


@singleton_object
class TrivialSpace(HilbertSpace, Expression, metaclass=Singleton):
    """The 'nullspace', i.e. a one dimensional Hilbert space, which is a factor
    space of every other Hilbert space.

    This is the Hilbert space of scalars.
    """

    def __hash__(self):
        return hash(self.__class__)

    @property
    def args(self):
        """Empty tuple (no arguments)"""
        return ()

    @property
    def _order_key(self):
        return KeyTuple((-2, '_'))

    @property
    def dimension(self):
        return 1

    @property
    def has_basis(self):
        """True, by definition (the basis is defined as
        :obj:`~qnet.algebra.state_algebra.TrivialKet`)"""
        return True

    @property
    def basis_states(self):
        """Yield the :obj:`~qnet.algebra.state_algebra.TrivialKet`"""
        from qnet.algebra.core.state_algebra import TrivialKet
        yield TrivialKet

    def basis_state(self, index_or_label):
        """Return the basis state with the given index 0 or label "0".

        All other indices or labels raise an exception.

        Raises:
            IndexError: if index is different from 0
            KeyError: if label is differnt from ""
        """
        from qnet.algebra.core.state_algebra import TrivialKet
        if index_or_label in [0, "0"]:
            return TrivialKet
        else:
            if isinstance(index_or_label, int):
                raise IndexError(
                    "No index %d in basis for TrivialSpace" % index_or_label)
            else:
                raise KeyError(
                    "No label %d in basis for TrivialSpace" % index_or_label)

    @property
    def basis_labels(self):
        """The one-element tuple containing the label '0'"""
        return tuple(["0", ])

    def remove(self, other):
        """Removing any Hilbert space from the trivial space yields the trivial
        space again"""
        return self

    def intersect(self, other):
        """The intersection of the trivial space with any other space is only
        the trivial space"""
        return self

    @property
    def local_factors(self):
        """Empty list (the trivial space has no factors)"""
        return ()

    def is_strict_subfactor_of(self, other):
        """The trivial space is a subfactor of any other space (except
        itself)"""
        if other is TrivialSpace:
            return False
        return True

    @property
    def label(self):
        return "null"


@singleton_object
class FullSpace(HilbertSpace, Expression, metaclass=Singleton):
    """The 'full space', i.e. a Hilbert space that includes any other Hilbert
    space as a tensor factor.

    The `FullSpace` has no defined basis, any related properties will raise
    :class:`.BasisNotSetError`
    """

    @property
    def args(self):
        """Empty tuple (no arguments)"""
        return ()

    def __hash__(self):
        return hash(self.__class__)

    @property
    def _order_key(self):
        return KeyTuple((-1, '_'))

    def remove(self, other):
        """Raise AlgebraError, as the remaining space is undefined"""
        raise AlgebraError("Cannot remove anything from FullSpace")

    @property
    def local_factors(self):
        """Raise AlgebraError, as the the factors of the full space are
        undefined"""
        raise AlgebraError("FullSpace has no local_factors")

    def intersect(self, other):
        """Return `other`"""
        return other

    def is_strict_subfactor_of(self, other):
        """False, as the full space by definition is not contained in any other
        space"""
        return False

    @property
    def label(self):
        return "total"


###############################################################################
# Algebra Operations
###############################################################################


class ProductSpace(HilbertSpace, Operation):
    """Tensor product of local Hilbert spaces

    >>> hs1 = LocalSpace('1', basis=(0,1))
    >>> hs2 = LocalSpace('2', basis=(0,1))
    >>> hs = hs1 * hs2
    >>> hs.basis_labels
    ('0,0', '0,1', '1,0', '1,1')
    """

    _neutral_element = TrivialSpace
    simplifications = [empty_trivial, assoc, convert_to_spaces, idem,
                       filter_neutral]

    def __init__(self, *local_spaces):
        if len(set(local_spaces)) != len(local_spaces):
            raise ValueError("Nondistinct spaces: %s" % repr(local_spaces))
        try:
            self._dimension = functools.reduce(
                    operator.mul,
                    [ls.dimension for ls in local_spaces], 1)
        except BasisNotSetError:
            self._dimension = None
        # determine the basis labels automatically
        try:
            ls_bases = [ls.basis_labels for ls in local_spaces]
            basis = []
            for label_tuple in cartesian_product(*ls_bases):
                basis.append(",".join([str(l) for l in label_tuple]))
            self._basis = tuple(basis)
        except BasisNotSetError:
            self._basis = None
        op_keys = [space._order_key for space in local_spaces]
        self._order_key = KeyTuple([v for op_key in op_keys for v in op_key])
        super().__init__(*local_spaces)  # Operation __init__

    @classmethod
    def create(cls, *local_spaces):
        if any(local_space is FullSpace for local_space in local_spaces):
            return FullSpace
        return super().create(*local_spaces)

    @property
    def has_basis(self):
        """True if the all the local factors of the `ProductSpace` have a
        defined basis"""
        return self._basis is not None

    @property
    def basis_states(self):
        """Yield an iterator over the states (:class:`.TensorKet` instances)
        that form the canonical basis of the Hilbert space

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        from qnet.algebra.core.state_algebra import BasisKet, TensorKet
        # importing locally avoids circular import
        ls_bases = [ls.basis_labels for ls in self.local_factors]
        for label_tuple in cartesian_product(*ls_bases):
            yield TensorKet(
                *[BasisKet(label, hs=ls) for (ls, label)
                  in zip(self.local_factors, label_tuple)])

    @property
    def basis_labels(self):
        """Tuple of basis labels. Each basis label consists of the labels of
        the :class:`.BasisKet` states that factor the basis state, separated by
        commas.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        if self._basis is None:
            raise BasisNotSetError(
                "Hilbert space %s has no defined basis" % str(self))
        return self._basis

    def basis_state(self, index_or_label):
        """Return the basis state with the given index or label.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
            IndexError: if there is no basis state with the given index
            KeyError: if there is not basis state with the given label
        """
        from qnet.algebra.core.state_algebra import BasisKet, TensorKet
        if isinstance(index_or_label, int):  # index
            ls_bases = [ls.basis_labels for ls in self.local_factors]
            label_tuple = list(cartesian_product(*ls_bases))[index_or_label]
            try:
                return TensorKet(
                    *[BasisKet(label, hs=ls) for (ls, label)
                        in zip(self.local_factors, label_tuple)])
            except ValueError as exc_info:
                raise IndexError(str(exc_info))
        else:  # label
            local_labels = index_or_label.split(",")
            if len(local_labels) != len(self.local_factors):
                raise KeyError(
                    "label %s for Hilbert space %s must be comma-separated "
                    "concatenation of local labels" % (index_or_label, self))
            try:
                return TensorKet(
                    *[BasisKet(label, hs=ls) for (ls, label)
                      in zip(self.local_factors, local_labels)])
            except ValueError as exc_info:
                raise KeyError(str(exc_info))

    @property
    def dimension(self):
        """Dimension of the Hilbert space.

        Raises:
            .BasisNotSetError: if the Hilbert space has no defined basis
        """
        if self._dimension is not None:
            return self._dimension
        else:
            raise BasisNotSetError(
                "Hilbert space %s has no defined basis" % str(self))

    def remove(self, other):
        """Remove a particular factor from a tensor product space."""
        if other is FullSpace:
            return TrivialSpace
        if other is TrivialSpace:
            return self
        if isinstance(other, ProductSpace):
            oops = set(other.operands)
        else:
            oops = {other}
        return ProductSpace.create(
                *sorted(set(self.operands).difference(oops)))

    @property
    def local_factors(self):
        """The :class:`LocalSpace` instances that make up the product"""
        return self.operands

    @classmethod
    def order_key(cls, obj):
        """Key by which operands are sorted"""
        assert isinstance(obj, HilbertSpace)
        return obj._order_key

    def intersect(self, other):
        """Find the mutual tensor factors of two Hilbert spaces."""
        if other is FullSpace:
            return self
        if other is TrivialSpace:
            return TrivialSpace
        if isinstance(other, ProductSpace):
            other_ops = set(other.operands)
        else:
            other_ops = {other}
        return ProductSpace.create(
                *sorted(set(self.operands).intersection(other_ops)))

    def is_strict_subfactor_of(self, other):
        """Test if a space is included within a larger tensor product space.
        Not ``True`` if ``self == other``."""
        if isinstance(other, ProductSpace):
            return set(self.operands) < set(other.operands)
        if other is FullSpace:
            return True
        return False
