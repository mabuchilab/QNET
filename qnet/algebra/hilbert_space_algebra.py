# coding=utf-8
#This file is part of QNET.
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
from abc import ABCMeta, abstractmethod
from itertools import product as cartesian_product

from qnet.algebra.abstract_algebra import (
        singleton, Expression, Operation, AlgebraError, tex, assoc, idem,
        filter_neutral)


class HilbertSpace(metaclass=ABCMeta):
    """Basic Hilbert space class from which concrete classes are derived."""

    _hilbert_tex_symbol = '\mathcal{H}'

    def tensor(self, other):
        """Tensor product between Hilbert spaces

        :param other: Other Hilbert space
        :type other: HilbertSpace
        :return: Tensor product space.
        :rtype: HilbertSpace
        """
        return ProductSpace.create(self, other)

    @abstractmethod
    def remove(self, other):
        """Remove a particular factor from a tensor product space."""
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def intersect(self, other):
        """Find the mutual tensor factors of two Hilbert spaces."""
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def local_factors(self):
        """Return tupe of LocalSpace objects that tensored together yield this
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
        """The full dimension of the Hilbert space"""
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

    @abstractmethod
    def is_strict_subfactor_of(self, other):
        """Test whether a Hilbert space occures as a strict sub-factor in
        (larger) Hilbert space"""
        raise NotImplementedError(self.__class__.__name__)

    def __len__(self):
        """The number of LocalSpace factors / degrees of freedom."""
        return len(self.local_factors())

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


@singleton
class TrivialSpace(HilbertSpace, Expression):
    """The 'nullspace', i.e. a one dimensional Hilbert space, which is a factor
    space of every other Hilbert space."""

    def __hash__(self):
        return hash(self.__class__)

    @property
    def args(self):
        """Empty tuple (no arguments)"""
        return ()

    def order_key(self):
        return "____"

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

    def tex(self):
        """Latex Representation of the trivial space"""
        return r"%s_{\rm null}" % self._hilbert_tex_symbol


@singleton
class FullSpace(HilbertSpace, Expression):
    """The 'full space', i.e. a Hilbert space that includes any other Hilbert
    space as a tensor factor."""

    @property
    def args(self):
        """Empty tuple (no arguments)"""
        return ()

    def __hash__(self):
        return hash(self.__class__)

    def order_key(self):
        return "____"

    def all_symbols(self):
        """Empty set (no symbols)"""
        return set(())

    def remove(self, other):
        """Raise AlgebraError, as the remaining space is undefined"""
        raise AlgebraError()

    def local_factors(self):
        """Raise AlgebraError, as the the factors of the full space are
        undefined"""
        raise AlgebraError()

    def intersect(self, other):
        """Return `other`"""
        return other

    def is_strict_subfactor_of(self, other):
        """False, as the full space by definition is not contained in any other
        space"""
        return False

    def __eq__(self, other):
        return self is other

    def tex(self):
        """Latex Representation of the full space"""
        return r"%s_{\rm total}" % self._hilbert_tex_symbol


class FiniteHilbertSpace(HilbertSpace, metaclass=ABCMeta):
    """Superclass for Hilbert spaces that can have a well-defined basis.

    Every instance of FiniteHilbertSpace must define a _registry_key that
    uniquely identifies the Hilbert space. Two instances that have the same key
    are considered identical, and are associated with the same basis.

    The basis must be set via the `basis` or `dimension` properties, or the
    `set_basis` method. Once a basis is set, the labels for the basis set may
    be overwritten at any time, as long as the dimension of the Hilbert space
    does not change. To change the dimension, the `set_basis` method must be
    called with the `force=True` argment.
    """

    _registry = {}  # dict _registry_key => tuple of basis labels

    _registry_key = None  # must be shadowd by subclass instances

    @property
    def basis(self):
        """Basis of the the Hilbert space, or None if no basis is set"""
        try:
            res = self._registry[self._registry_key]
        except KeyError:
            res = None
        return res

    @basis.setter
    def basis(self, basis):
        self.set_basis(basis, force=False)

    def set_basis(self, basis, force=False):
        """Set the basis states for the Hilbert space (as a list of labels).
        If a basis of a different dimension has previously been registered for
        the same Hilbert space, `force=True` must be given, or a
        ChangingDimensionError` is raised.
        """
        if basis is None:
            return
        if not force and self._registry_key in self._registry:
            if self.dimension != len(basis):
                raise ChangingDimensionError(
                        "The Hilbert space already has a basis set for "
                        "To change the dimension, the `set_basis` method "
                        "must be used with `force=True`")
        self._registry[self._registry_key] = tuple(basis)

    def get_basis(self, raise_basis_not_set_error=True):
        """Return the `basis` property, but if `raise_basis_not_set_error` is
        True, raise a `BasisNotSetError` if no basis is set, instead of
        returning None"""
        basis = self.basis
        if basis is None:
            if raise_basis_not_set_error:
                raise BasisNotSetError("Hilbert space %s has no defined basis")
        return basis

    @property
    def dimension(self):
        """The full dimension of the Hilbert space, or None if no basis is
        set"""
        basis = self.basis
        if basis is None:
            return None
        else:
            return len(basis)

    @dimension.setter
    def dimension(self, dimension):
        known_dimension = self.dimension
        if dimension != known_dimension:
            if known_dimension is None:
                self.basis = list(range(dimension))
            else:
                raise ChangingDimensionError((
                        "The Hilbert space already has a basis set for "
                        "dimension %s. To change the dimension, the "
                        "`set_basis` method must be used with `force=True`")
                        % (known_dimension))

    def __hash__(self):
        return hash((self.__class__, self._registry_key))

    def __eq__(self, other):
        try:
            return self._registry_key == other._registry_key != None
        except AttributeError:
            return False


class LocalSpace(FiniteHilbertSpace, Expression):
    """A local Hilbert space, i.e., for a single degree of freedom.

    Args:
        name (str): Name (subscript) of the Hilbert space
        basis (tuple or None): Set an explicit basis for the Hilbert space
            (tuple of labels for the basis states)
        dimension (int or None): Specify the dimension $n$ of the Hilbert
            space.  This implies a basis numbered from 0  to $n-1$.

    If no `basis` or `dimension` is specified during initialization, it may be
    set later by assigning to the corresponding properties. Note that the
    `basis` and `dimension` arguments are mutually exclusive.
    """

    def __init__(self, name, *, basis=None, dimension=None):
        name = str(name)
        self._registry_key = name
        if basis is None:
            if dimension is not None:
                self.basis = list(range(int(dimension)))
        else:
            if dimension is not None:
                if dimension != len(basis):
                    raise ValueError("basis and dimension are incompatible")
            self.basis = basis
        self.name = name

    @property
    def args(self):
        return (self.name, )

    @property
    def kwargs(self):
        basis = self.basis
        dim = self.dimension
        if basis is not None:
            if basis == tuple(range(int(dim))):
                return {'dimension': dim}
            else:
                return {'basis': basis}
        else:
            return {}

    def all_symbols(self):
        return {}

    def order_key(self):
        return self.name

    def remove(self, other):
        if other == self:
            return TrivialSpace
        return self

    def intersect(self, other):
        if self in other.local_factors():
            return self
        return TrivialSpace

    def local_factors(self):
        return self,

    def is_strict_subfactor_of(self, other):
        if isinstance(other, ProductSpace) and self in other.operands:
            assert len(other.operands) > 1
            return True
        if other is FullSpace:
            return True
        return False

    def tex(self):
        """TeX representation of the Local Hilbert space"""
        return "%s_{%s}" % (self._hilbert_tex_symbol, tex(self.name))

    def __str__(self):
        return "H_{}".format(self.name)


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


class ProductSpace(FiniteHilbertSpace, Operation):
    """Tensor product space class for an arbitrary number of LocalSpace
    factors."""

    signature = (HilbertSpace, '*'), {}
    neutral_element = TrivialSpace
    _simplifications = [empty_trivial, assoc, convert_to_spaces, idem,
                        filter_neutral]

    def __init__(self, *local_spaces):
        if len(set(local_spaces)) != len(local_spaces):
            raise ValueError(repr(local_spaces))
        self._registry_key = local_spaces
        super().__init__(*local_spaces)

    @classmethod
    def create(cls, *local_spaces):
        if any(local_space is FullSpace for local_space in local_spaces):
            return FullSpace
        return super().create(*local_spaces)

    @property
    def basis(self):
        """Basis of the ProductSpace. An explicit basis may be set by assigning
        to the basis property. If no basis is set explicitly, the the basis is
        obtained from the bases of the factors. The basis labels are
        comma-separated combinations of the basis labels of the factors::

        >>> hs1 = LocalSpace('tls', '1', basis=(0,1))
        >>> hs2 = LocalSpace('tls', '2', basis=(0,1))
        >>> hs = hs1 * hs2
        >>> hs.basis
        ('0,0', '0,1', '1,0', '1,1')
        >>> hs.basis = ('00', '01', '10', '11')
        >>> hs.basis
        ('00', '01', '10', '11')
        """
        bases = []
        dimension = self.dimension
        if dimension is None:
            return None
        result = super().basis  # look for explicit basis in registry
        if result is None:
            # If no explicit basis set, delegate to operands
            result = []
            for op in self.operands:
                op_basis = op.basis
                if op_basis is None:
                    return None
                else:
                    bases.append(op.basis)
            for label_tuple in cartesian_product(*bases):
                result.append(",".join([str(l) for l in label_tuple]))
            return result
        else:
            if len(result) != dimension:
                self.dimension = dimension  # raise ChangingDimensionError
            return result

    @property
    def dimension(self):
        """Dimension of the Hilbert space"""
        dimension = 1
        for op in self.operands:
            if op.dimension is None:
                return None
            dimension *= op.dimension
        return dimension

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

    def tex(self):
        return r' \otimes '.join([tex(op) for op in self.operands])

    def __str__(self):
        return r' * '.join([tex(op) for op in self.operands])


class BasisNotSetError(AlgebraError):
    """Raised if the basis or Hilbert space dimension is requested without a
    basis being set"""


class ChangingDimensionError(AlgebraError):
    """Raised if the dimension of a previously set basis is changed"""
