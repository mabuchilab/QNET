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
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

r"""
The Hilbert Space Algebra
=========================

This module defines some simple classes to describe simple and
*composite/tensor* (i.e., multiple degree of freedom)
Hilbert spaces of quantum systems.

For more details see :ref:`hilbert_space_algebra`.
"""
import re
import operator
import functools
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import product as cartesian_product

from .abstract_algebra import (
        Expression, Operation, AlgebraError, assoc, idem, filter_neutral,
        cache_attr)
from .singleton import Singleton, singleton_object


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
        """The full dimension of the Hilbert space (or None) if the dimension
        is not known"""
        return None

    def get_dimension(self, raise_basis_not_set_error=True):
        """Return the `dimension` property, but if `raise_basis_not_set_error`
        is True, raise a `BasisNotSetError` if no basis is set, instead of
        returning None"""
        dimension = self.dimension
        if dimension is None:
            if raise_basis_not_set_error:
                raise BasisNotSetError("Hilbert space %s has no defined basis")
        return dimension

    @property
    def basis(self):
        """Basis of the the Hilbert space, or None if no basis is set"""
        return None

    def get_basis(self, raise_basis_not_set_error=True):
        """Return the `basis` property, but if `raise_basis_not_set_error` is
        True, raise a `BasisNotSetError` if no basis is set, instead of
        returning None"""
        basis = self.basis
        if basis is None:
            if raise_basis_not_set_error:
                raise BasisNotSetError("Hilbert space %s has no defined basis")
        return basis

    @abstractmethod
    def is_strict_subfactor_of(self, other):
        """Test whether a Hilbert space occures as a strict sub-factor in
        (larger) Hilbert space"""
        raise NotImplementedError(self.__class__.__name__)

    def _render(self, fmt, adjoint=False):
        assert not adjoint, "adjoint not defined"
        printer = getattr(self, "_"+fmt+"_printer")
        return printer.hilbert_space_fmt.format(
                label=printer.render_hs_label(self))

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

    If no `basis` or `dimension` is specified during initialization, it may be
    set later by assigning to the corresponding properties. Note that the
    `basis` and `dimension` arguments are mutually exclusive.
    """
    _rx_label = re.compile('^[A-Za-z0-9.+-]+(_[A-Za-z0-9().+-]+)?$')

    def __init__(self, label, *, basis=None, dimension=None):
        label = str(label)
        if not self._rx_label.match(label):
            raise ValueError("label '%s' does not match pattern '%s'"
                             % (label, self._rx_label.pattern))
        self._basis = None
        self._dimension = None
        if basis is None:
            if dimension is not None:
                self._basis = tuple(range(int(dimension)))
                self._dimension = int(dimension)
        else:
            if dimension is not None:
                if dimension != len(basis):
                    raise ValueError("basis and dimension are incompatible")
            self._basis = tuple(basis)
            self._dimension = len(basis)
        self._label = label
        super().__init__(label, basis=basis, dimension=dimension)

    @property
    def args(self):
        return (self._label, )

    @property
    def label(self):
        """Label of the Hilbert space"""
        return self._label

    @property
    def basis(self):
        return self._basis

    @property
    def dimension(self):
        return self._dimension

    @property
    def kwargs(self):
        return {'basis': self._basis, 'dimension': self._dimension}

    def all_symbols(self):
        return {}

    def _order_key(self):
        return self._label

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

    def _order_key(self):
        return "____"

    @property
    def dimension(self):
        return 1

    @property
    def basis(self):
        return ("empty", )

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

    def _render(self, fmt, adjoint=False):
        printer = getattr(self, "_"+fmt+"_printer")
        return printer.hilbert_space_fmt.format(
                label=printer.render_string(self.label))


@singleton_object
class FullSpace(HilbertSpace, Expression, metaclass=Singleton):
    """The 'full space', i.e. a Hilbert space that includes any other Hilbert
    space as a tensor factor."""

    @property
    def args(self):
        """Empty tuple (no arguments)"""
        return ()

    def __hash__(self):
        return hash(self.__class__)

    def _order_key(self, a):
        return "____"

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

    def _render(self, fmt, adjoint=False):
        printer = getattr(self, "_"+fmt+"_printer")
        return printer.hilbert_space_fmt.format(
                label=printer.render_string(self.label))


###############################################################################
# Algebra Operations
###############################################################################


class ProductSpace(HilbertSpace, Operation):
    """Tensor product space class for an arbitrary number of LocalSpace
    factors.

    >>> hs1 = LocalSpace('1', basis=(0,1))
    >>> hs2 = LocalSpace('2', basis=(0,1))
    >>> hs = hs1 * hs2
    >>> hs.basis
    ('0,0', '0,1', '1,0', '1,1')
    """

    signature = (HilbertSpace, '*'), {}
    neutral_element = TrivialSpace
    _simplifications = [empty_trivial, assoc, convert_to_spaces, idem,
                        filter_neutral]

    def __init__(self, *local_spaces):
        if len(set(local_spaces)) != len(local_spaces):
            raise ValueError("Nondistinct spaces: %s" % repr(local_spaces))
        try:
            self._dimension = functools.reduce(
                    operator.mul,
                    [ls.get_dimension() for ls in local_spaces], 1)
        except BasisNotSetError:
            self._dimension = None
        # determine the basis automatically
        try:
            ls_bases = [ls.get_basis() for ls in local_spaces]
            basis = []
            for label_tuple in cartesian_product(*ls_bases):
                basis.append(",".join([str(l) for l in label_tuple]))
            self._basis = tuple(basis)
        except BasisNotSetError:
            self._basis = None
        super().__init__(*local_spaces)  # Operation __init__

    @classmethod
    def create(cls, *local_spaces):
        if any(local_space is FullSpace for local_space in local_spaces):
            return FullSpace
        return super().create(*local_spaces)

    @property
    def basis(self):
        """Basis of the ProductSpace, from the bases of the operands"""
        return self._basis

    @property
    def dimension(self):
        return self._dimension

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
        """The LocalSpace instances that make up the product"""
        return self.operands

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

    def _render(self, fmt, adjoint=False):
        assert not adjoint, "adjoint not defined"
        printer = getattr(self, "_"+fmt+"_printer")
        return printer.render_infix(self.operands, 'tensor_sym')
