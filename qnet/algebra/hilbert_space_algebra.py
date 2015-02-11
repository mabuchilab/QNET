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

This module defines some simple classes to describe simple and *composite/tensor* (i.e., multiple degree of freedom)
Hilbert spaces of quantum systems.

For more details see :ref:`hilbert_space_algebra`.
"""


from qnet.algebra.abstract_algebra import *

@six.add_metaclass(ABCMeta)
class HilbertSpace(object):
    """
    Basic Hilbert space class from which concrete classes are derived.
    """

    def tensor(self, other):
        """
        Tensor product between Hilbert spaces

        :param other: Other Hilbert space
        :type other: HilbertSpace
        :return: Tensor product space.
        :rtype: HilbertSpace
        """
        return ProductSpace.create(self, other)

    def remove(self, other):
        """
        Remove a particular factor from a tensor product space.

        :param other: Space to remove
        :type other: HilbertSpace
        :return: Hilbert space for remaining degrees of freedom.
        :rtype: HilbertSpace
        """
        return self._remove(other)

    @abstractmethod
    def _remove(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def intersect(self, other):
        """
        Find the mutual tensor *factors* of two Hilbert spaces.

        :param other: Other Hilbert space
        :type other: HilbertSpace
        """
        return self._intersect(other)

    @abstractmethod
    def _intersect(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def local_factors(self):
        """
        :return: A sequence of LocalSpace objects that tensored together yield this Hilbert space.
        :rtype: tuple of LocalSpace objects
        """
        return self._local_factors()

    @abstractmethod
    def _local_factors(self):
        raise NotImplementedError(self.__class__.__name__)

    def is_tensor_factor_of(self, other):
        """
        Test if a space is included within a larger tensor product space. Also ``True`` if ``self == other``.

        :param other: Other Hilbert space
        :type other: HilbertSpace
        :rtype: bool
        """
        return self <= other

    def is_strict_tensor_factor_of(self, other):
        """
        Test if a space is included within a larger tensor product space. Not ``True`` if ``self == other``.

        :param other: Other Hilbert space
        :type other: HilbertSpace
        :rtype: bool
        """
        return self < other

    @property
    def dimension(self):
        """
        :return: The full dimension of the Hilbert space
        :rtype: int
        """
        #noinspection PyTypeChecker,PyCallByClass,PyArgumentList
        return BasisRegistry.dimension(self)

    def is_strict_subfactor_of(self, other):
        """
        Test whether a Hilbert space occures as a strict sub-factor in (larger) Hilbert space
        :type other: HilbertSpace
        """
        return self._is_strict_subfactor_of(other)

    @abstractmethod
    def _is_strict_subfactor_of(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def __len__(self):
        """
        :return: The number of LocalSpace factors / degrees of freedom.
        :rtype: int
        """
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
    """
    The 'nullspace', i.e. a one dimensional Hilbert space, which is a factor space of every other Hilbert space.
    """

    def __hash__(self):
        return hash(self.__class__)

    def _order_key(self):
        return "____",

#    def tensor(self, other):
#        return other

    def _all_symbols(self):
        return set(())

    def _remove(self, other):
        return self

    def _intersect(self, other):
        return self

    def _local_factors(self):
        return ()

    def _is_strict_subfactor_of(self, other):
        if other is TrivialSpace:
            return False
        return True

    def __eq__(self, other):
        return self is other

    def _tex(self):
        return r"\mathcal{H}_{\rm null}"





@singleton
class FullSpace(HilbertSpace, Expression):
    """
    The 'full space', i.e. a Hilbert space, includes any other Hilbert space as a tensor factor.
    """

    def __hash__(self):
        return hash(self.__class__)

    def _order_key(self):
        return "____",

#    def tensor(self, other):
#        return self

    def _all_symbols(self):
        return set(())

    def _remove(self, other):
        raise AlgebraError()

    def _local_factors(self):
        raise AlgebraError()

    def _intersect(self, other):
        return other

    def _is_strict_subfactor_of(self, other):
        return False

    def __eq__(self, other):
        return self is other

    def _tex(self):
        return r"\mathcal{H}_{\rm total}"




@check_signature
class LocalSpace(HilbertSpace, Operation):
    """
    Basic class to instantiate a local Hilbert space, i.e., for a single degree of freedom.

        ``LocalSpace(name, namespace)``

    :param name: The identifier of the local space / degree of freedom
    :type name: str
    :param namespace: The namespace for the degree of freedom, useful in hierarchical system models.
    :type namespace: str
    """
    signature = basestring, basestring

    def _order_key(self):
        return self.operands

    def _remove(self, other):
        if other == self:
            return TrivialSpace
        return self

    def _intersect(self, other):
        if self in other.local_factors():
            return self
        return TrivialSpace

    def _local_factors(self):
        return self,

    def _is_strict_subfactor_of(self, other):
        if isinstance(other, ProductSpace) and self in other.operands:
            #noinspection PyTypeChecker
            assert len(other.operands) > 1
            return True
        if other is FullSpace:
            return True
        return False

    def _get_dimension(self):
        return BasisRegistry.dimension(self)

    def _set_dimension(self, dimension):
        try:
            current_basis = list(self.basis)
            current_length = len(current_basis)
            if current_length == dimension:
                return
            if current_basis != range(current_length):
                raise ValueError('It appears that the current basis of {} is not simply a range of integer-labelled states: {}'.format(str(self), str(current_basis)))
            BasisRegistry.set_basis(self, range(dimension))
        except BasisNotSetError:
            BasisRegistry.set_basis(self, range(dimension))

    dimension = property(_get_dimension, _set_dimension, doc="The local state space dimension.")

    @property
    def basis(self):
        """
        :return: The set of basis states of the local Hilbert space
        :rtype: sequence of int or str
        """
        #noinspection PyCallByClass,PyArgumentList,PyTypeChecker
        return BasisRegistry.get_basis(self)

    def _tex(self):
        name, namespace = self.operands
        if namespace:
            return "{{{}}}_{{{}}}".format(tex(name), tex(namespace))
        return "{{{}}}".format(tex(name))

    def __str__(self):
        name, namespace = self.operands
        if namespace:
            return "{}_{}".format(name, namespace)
        return name


#
#noinspection PyRedeclaration
def local_space(name, namespace = "", dimension = None, basis = None):
    """
    Create a LocalSpace with by default empty namespace string.
    If one also provides a set of basis states, these get stored via the BasisRegistry object.
    ALternatively, one may provide a dimension such that the states are simply labeled by a range of integers:

        ``[0, 1, 2, ..., dimension -1]``

    :param name: Local space identifier
    :type name: str or int
    :param namespace: Local space namespace, see LocalSpace documentation
    :type namespace: str
    :param dimension: Dimension of local space (optional)
    :type dimension: int
    :param basis: Basis state labels for local space
    :type basis: sequence of int or sequence of str
    """
    if isinstance(name, int):
        s = LocalSpace.create(str(name), namespace)
    else:
        s = LocalSpace.create(name, namespace)
    if dimension:
        if basis:
            assert len(basis) == dimension
        else:
            basis = range(dimension)
    if basis:
        #noinspection PyArgumentList,PyTypeChecker
        BasisRegistry.set_basis(s, basis)
    return  s






def convert_to_spaces_mtd(dcls, clsmtd, cls, *ops):
    """
    For all operands that are merely of type str or int, substitute LocalSpace objects with corresponding labels:
    For a string, just itself, for an int, a string version of that int.
    """
    cops = [o if isinstance(o, HilbertSpace) else local_space(o) for o in ops]
    return clsmtd(cls, *cops)
convert_to_spaces = preprocess_create_with(convert_to_spaces_mtd)

@assoc
@convert_to_spaces
@idem
@check_signature_assoc
@filter_neutral
class ProductSpace(HilbertSpace, Operation):
    """
    Tensor product space class for an arbitrary number of local space factors.

        ``ProductSpace(*factor_spaces)``

    :param factor_spaces: The Hilbert spaces to be tensored together.
    :type factor_spaces: HilbertSpace
    """

    signature = HilbertSpace,
    neutral_element = TrivialSpace

    @classmethod
    def create(cls, *operands):
        if any(o is FullSpace for o in operands):
            return FullSpace
        return cls(*operands)

    def _remove(self, other):
        if other is FullSpace:
            return TrivialSpace
        if other is TrivialSpace:
            return self
        if isinstance(other, ProductSpace):
            oops = set(other.operands)
        else:
            oops = {other}
        return ProductSpace.create(*sorted(set(self.operands).difference(oops)))

    def _local_factors(self):
        return self.operands

    def _intersect(self, other):
        if other is FullSpace:
            return self
        if other is TrivialSpace:
            return TrivialSpace
        if isinstance(other, ProductSpace):
            other_ops = set(other.operands)
        else:
            other_ops = {other}
        return ProductSpace.create(*sorted(set(self.operands).intersection(other_ops)))

    def _is_strict_subfactor_of(self, other):
        if isinstance(other, ProductSpace):
            return set(self.operands) < set(other.operands)
        if other is FullSpace:
            return True
        return False


    def _tex(self):
        return " \otimes ".join(map(tex, self.operands))

class BasisNotSetError(AlgebraError):
    """
    Is raised when the basis states of a LocalSpace are requested before being defined.

    :param local_space:
    :type local_space:
    """

    def __init__(self, local_space):
        msg = """The basis for the local space {0!s} has not been set yet.
Please set the basis states via the command:
    qnet.algebra.HilbertSpaceAlgebra.BasisRegistry.set_basis({0!r}, basis)""".format(local_space)
        super(BasisNotSetError, self).__init__(msg)

@singleton
class BasisRegistry(object):
    """
    Singleton class to store information about the basis states of all
    """
    registry = {}

    def set_basis(self, local_space, basis):
        """
        Set the basis states of a local Hilbert space.

        :param local_space: Local Hilbert space object
        :type local_space: LocalSpace
        :param basis: A sequence of state labels
        :type basis: sequence of int or sequence of str
        """
        previous = self.registry.get(local_space, basis)
        if basis != previous:
            print(
            "Warning: changing basis specification for registered LocalSpace {!s} from {!r} to {!r}".format(local_space,
                previous, basis))
        self.registry[local_space] = basis

    def get_basis(self, local_space):
        """
        Retrieve the basis states of a local Hilbert space.
        If no basis states have been set, raise a BasisNotSetError exception.

        :param local_space: Local Hilbert space object
        :type local_space: LocalSpace
        :return: A sequence of state labels
        :rtype: sequence of int or str
        :raise: BasisNotSetError
        """
        if local_space in self.registry:
            return self.registry[local_space]
        raise BasisNotSetError(local_space)

    def dimension(self, space):
        """
        Compute the full dimension of a Hilbert space object.

        :param space: Hilbert space to compute dimension for
        :type space: HilbertSpace
        :return: Dimension of space
        :rtype: int
        """
        if space is TrivialSpace:
            return 1
        if space is FullSpace:
            return inf
        dims = [len(self.get_basis(s)) for s in space.local_factors()]
        return prod(dims)

