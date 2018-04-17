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

r"""
This module defines some simple classes to describe simple and
*composite/tensor* (i.e., multiple degree of freedom)
Hilbert spaces of quantum systems.

For more details see :ref:`hilbert_space_algebra`.
"""
import re
import operator
import functools
from collections import OrderedDict
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import product as cartesian_product

from .abstract_algebra import (
        Expression, Operation, AlgebraError, assoc, idem, filter_neutral)
from .ordering import KeyTuple
from .singleton import Singleton, singleton_object

__all__ = [
    'BasisNotSetError', 'HilbertSpace', 'LocalSpace', 'ProductSpace',
    'FullSpace', 'TrivialSpace']

__private__ = [  # anything not in __all__ must be in __private__
    'convert_to_spaces', 'empty_trivial']


###############################################################################
# Exceptions
###############################################################################


class BasisNotSetError(AlgebraError):
    """Raised if the basis or a Hilbert space dimension is requested but is not
    available"""


###############################################################################
# Algebraic properties
###############################################################################


def convert_to_spaces(cls, ops, kwargs):
    """For all operands that are merely of type str or int, substitute
    LocalSpace objects with corresponding labels:
    For a string, just itself, for an int, a string version of that int.
    """
    cops = [o if isinstance(o, HilbertSpace) else LocalSpace(o) for o in ops]
    return cops, kwargs


def empty_trivial(cls, ops, kwargs):
    """A ProductSpace of zero Hilbert spaces should yield the TrivialSpace"""
    if len(ops) == 0:
        return TrivialSpace
    else:
        return ops, kwargs


###############################################################################
# Abstract base classes
###############################################################################


class HilbertSpace(metaclass=ABCMeta):
    """Basic Hilbert space class from which concrete classes are derived."""

    def tensor(self, *others):
        """Tensor product between Hilbert spaces

        :param others: Other Hilbert space(s)
        :type others: HilbertSpace
        :return: Tensor product space.
        :rtype: HilbertSpace
        """
        return ProductSpace.create(self, *others)

    @abstractmethod
    def remove(self, other):
        """Remove a particular factor from a tensor product space."""
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def intersect(self, other):
        """Find the mutual tensor factors of two Hilbert spaces."""
        raise NotImplementedError(self.__class__.__name__)

    @abstractproperty
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
            BasisNotSetError: if the Hilbert space has no defined basis
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    @property
    def has_basis(self):
        """True if the Hilbert space has a basis"""
        return False

    @property
    def basis_states(self):
        """Yield an iterator over the states (:class:`Ket` instances) that form
        the canonical basis of the Hilbert space

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    def basis_state(self, index_or_label):
        """Return the basis state with the given index or label.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
            IndexError: if there is no basis state with the given index
            KeyError: if there is not basis state with the given label
        """
        raise BasisNotSetError(
            "Hilbert space %s has no defined basis" % str(self))

    @property
    def basis_labels(self):
        """Tuple of basis labels.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
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

    def __div__(self, other):
        return self.remove(other)

    __truediv__ = __div__

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
    """A local Hilbert space, i.e., for a single degree of freedom.

    Args:
        label (str): label (subscript) of the Hilbert space
        basis (tuple or None): Set an explicit basis for the Hilbert space
            (tuple of labels for the basis states)
        dimension (int or None): Specify the dimension $n$ of the Hilbert
            space.  This implies a basis numbered from 0  to $n-1$.
        local_identifiers (dict): Mapping of class names of
            :class:`~qnet.algebra.operator_algebra.LocalOperator` subclasses
            to identifier names. Used e.g. 'b' instead of the default 'a' for
            the anihilation operator. This can be a dict or a dict-compatible
            structure, e.g. a list/tuple of key-value tuples.
        order_index (int or None): An optional key that determines the
            preferred order of Hilbert spaces. This also changes the order of
            e.g. sums or products of Operators. Hilbert spaces will be ordered
            from left to right be increasing `order_index`; Hilbert spaces
            without an explicit `order_index` are sorted by their label
    """
    _rx_label = re.compile('^[A-Za-z0-9.+-]+(_[A-Za-z0-9().+-]+)?$')

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
            sorted_local_identifiers = tuple(
                sorted(tuple(local_identifiers.items())))
        except TypeError:
            # this will happen e.g. if the keys in local_identifier are types
            # instead of class names
            raise TypeError(
                "local_identifier must map class names to identifier strings")

        label = str(label)
        if not self._rx_label.match(label):
            raise ValueError("label '%s' does not match pattern '%s'"
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
            sorted_local_identifiers))
        self._basis = basis
        self._dimension = dimension
        self._local_identifiers = local_identifiers
        self._order_index = order_index
        self._kwargs = OrderedDict([
            ('basis', self._basis), ('dimension', self._dimension),
            ('local_identifiers', sorted_local_identifiers),
            ('order_index', self._order_index)])
        self._minimal_kwargs = self._kwargs.copy()
        for key in default_args:
            del self._minimal_kwargs[key]

        super().__init__(
            label, basis=basis, dimension=dimension,
            local_identifiers=sorted_local_identifiers,
            order_index=order_index)

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
        """Yield an iterator over the states (:class:`BasisKet` instances) that
        form the canonical basis of the Hilbert space

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
        """
        from qnet.algebra.state_algebra import BasisKet  # avoid circ. import
        for label in self.basis_labels:
            yield BasisKet(label, hs=self)

    def basis_state(self, index_or_label):
        """Return the basis state with the given index or label.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
            IndexError: if there is no basis state with the given index
            KeyError: if there is not basis state with the given label
        """
        from qnet.algebra.state_algebra import BasisKet  # avoid circ. import
        try:
            return BasisKet(index_or_label, hs=self)
        except ValueError as exc_info:
            if isinstance(index_or_label, int):
                raise IndexError(str(exc_info))
            else:
                raise KeyError(str(exc_info))

    @property
    def basis_labels(self):
        """Tuple of basis labels.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
        """
        if self._basis is None:
            raise BasisNotSetError(
                "Hilbert space %s has no defined basis" % str(self))
        return self._basis

    @property
    def dimension(self):
        """Dimension of the Hilbert space.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
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

    def all_symbols(self):
        """Empty list"""
        return {}

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
            label_or_index (int or str): If `int`, the index of a basis state;
                if `str`, the label of a basis state
            n (int): The increment

        Raises:
            IndexError: If going beyond the last or first basis state
            ValueError: If `label` is not a label for any basis state in the
                Hilbert space
            BasisNotSetError: If the Hilbert space has no defined basis
            TypeError: if `label_or_index` is neither a `str` nor an `int`
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


@singleton_object
class TrivialSpace(HilbertSpace, Expression, metaclass=Singleton):
    """The 'nullspace', i.e. a one dimensional Hilbert space, which is a factor
    space of every other Hilbert space."""

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
        from qnet.algebra.state_algebra import TrivialKet
        yield TrivialKet

    def basis_state(self, index_or_label):
        """Return the basis state with the given index 0 or label "0".

        All other indices or labels raise an exception.

        Raises:
            IndexError: if index is different from 0
            KeyError: if label is differnt from ""
        """
        from qnet.algebra.state_algebra import TrivialKet
        if index_or_label in [0, "0"]:
            return TrivialKet
        else:
            if isinstance(index_or_label, int):
                raise IndexError("No index %d in basis for TrivialSpace"
                                 % index_or_label)
            else:
                raise KeyError("No label %d in basis for TrivialSpace"
                                % index_or_label)

    @property
    def basis_labels(self):
        """The one-element tuple containing the label '0'"""
        return tuple(["0", ])

    def all_symbols(self):
        """Empty set (no symbols)"""
        return set(())

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

    def __eq__(self, other):
        return self is other

    @property
    def label(self):
        return "null"


@singleton_object
class FullSpace(HilbertSpace, Expression, metaclass=Singleton):
    """The 'full space', i.e. a Hilbert space that includes any other Hilbert
    space as a tensor factor.

    The `FullSpace` has no defined basis, any related properties will raise
    :class:`BasisNotSetError`
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

    def all_symbols(self):
        """Empty set (no symbols)"""
        return set(())

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

    def __eq__(self, other):
        return self is other

    @property
    def label(self):
        return "total"


###############################################################################
# Algebra Operations
###############################################################################


class ProductSpace(HilbertSpace, Operation):
    """Tensor product space class for an arbitrary number of
    :class:`LocalSpace` factors.

    >>> hs1 = LocalSpace('1', basis=(0,1))
    >>> hs2 = LocalSpace('2', basis=(0,1))
    >>> hs = hs1 * hs2
    >>> hs.basis_labels
    ('0,0', '0,1', '1,0', '1,1')
    """

    neutral_element = TrivialSpace
    _simplifications = [empty_trivial, assoc, convert_to_spaces, idem,
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
        self._has_basis = all([ls.has_basis for ls in local_spaces])
        self._basis = None  # delayed until first call to basis_labels()
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
        return self._has_basis

    @property
    def basis_states(self):
        """Yield an iterator over the states (:class:`TensorKet` instances)
        that form the canonical basis of the Hilbert space

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
        """
        from qnet.algebra.state_algebra import BasisKet, TensorKet
        # importing locally avoids circular import
        ls_bases = [ls.basis_labels for ls in self.local_factors]
        for label_tuple in cartesian_product(*ls_bases):
            yield TensorKet(
                *[BasisKet(label, hs=ls) for (ls, label)
                  in zip(self.local_factors, label_tuple)])

    @property
    def basis_labels(self):
        """Tuple of basis labels. Each basis label consists of the labels of
        the :class:`BasisKet` states that factor the basis state, separated by
        commas.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
        """
        if self._has_basis:
            if self._basis is None:
                # Calculating the basis for a Product space can be very
                # expensive, which is why we delay calculating self._basis for
                # as long as possible
                ls_bases = [ls.basis_labels for ls in self.args]
                basis = []
                for label_tuple in cartesian_product(*ls_bases):
                    basis.append(",".join(label_tuple))
                self._basis = tuple(basis)
            return self._basis
        else:
            raise BasisNotSetError(
                "Hilbert space %s has no defined basis" % str(self))

    def basis_state(self, index_or_label):
        """Return the basis state with the given index or label.

        Raises:
            BasisNotSetError: if the Hilbert space has no defined basis
            IndexError: if there is no basis state with the given index
            KeyError: if there is not basis state with the given label
        """
        from qnet.algebra.state_algebra import BasisKet, TensorKet
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
            BasisNotSetError: if the Hilbert space has no defined basis
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

