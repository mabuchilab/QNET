#!/usr/bin/env python
# encoding: utf-8

from qnet.algebra.abstract_algebra import *


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

    def _remove(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def intersect(self, other):
        """
        Find the mutual tensor *factors* of two Hilbert spaces,
        :param other: Other Hilbert space
        :type other: HilbertSpace
        """
        return self._intersect(other)

    def _intersect(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def local_factors(self):
        """
        :return: A sequence of LocalSpace objects that tensored together yield this Hilbert space.
        :rtype: tuple of LocalSpace objects
        """
        return self._local_factors()

    def _local_factors(self):
        raise NotImplementedError(self.__class__.__name__)

    def is_tensor_factor_of(self, other):
        """
        Test if a space is included within a larger tensor product space. Also True if self == other.
        :param other: Other Hilbert space
        :type other: HilbertSpace
        :rtype: bool
        """
        return self <= other

    def is_strict_tensor_factor_of(self, other):
        """
        Test if a space is included within a larger tensor product space. Not True if self == other.
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
        return BasisRegistry.dimension(self)

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

    def __and__(self, other):
        return self.intersect(other)

    def __lt__(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def __gt__(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return self == other or self > other






def singleton(cls):
    """
    Singleton class decorator. Turns a class object into a unique instance.
    :param cls: Class to decorate
    :type cls: type
    :return: The singleton instance of that class
    :rtype: cls
    """
    s = cls()
    s.__call__ = lambda : s
    return s

@singleton
class TrivialSpace(HilbertSpace):
    """
    The 'nullspace', i.e. a one dimensional Hilbert space, which is a factor space of every other Hilbert space.
    """

    def tensor(self, other):
        return other

    def _remove(self, other):
        return self

    def _intersect(self, other):
        return self

    def _local_factors(self):
        return ()

    def __hash__(self):
        return hash(self.__class__)

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return self is other





@singleton
class FullSpace(HilbertSpace):
    """
    The 'full space', i.e. a Hilbert space, includes any other Hilbert space as a tensor factor.
    """

    def tensor(self, other):
        return self

    def _remove(self, other):
        raise AlgebraError()

    def _local_factors(self):
        raise AlgebraError()

    def _intersect(self, other):
        return other


    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return self is other

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True





@check_signature
class LocalSpace(HilbertSpace, Operation):
    """
    Basic class to instantiate a local Hilbert space, i.e., for a single degree of freedom.
        LocalSpace(name, namespace)
    :param name: The identifier of the local space / degree of freedom
    :type name: str
    :param namespace: The namespace for the degree of freedom, useful in hierarchical system models.
    :type namespace: str
    """
    signature = str, str


    def _remove(self, other):
        if other == self:
            return TrivialSpace
        return self

    def _intersect(self, other):
        if other == self:
            return self
        return TrivialSpace

    def _local_factors(self):
        return (self,)

    def __lt__(self, other):
        if not isinstance(other, HilbertSpace):
            return NotImplemented
        if isinstance(other, ProductSpace) and self in other.operands:
            return True
        if other is FullSpace:
            return True
        return False

    def __gt__(self, other):
        if not isinstance(other, HilbertSpace):
            return NotImplemented
        if other is TrivialSpace:
            return True
        return False

    @property
    def basis(self):
        """
        :return: The set of basis states of the local Hilbert space
        :rtype: sequence of int or str
        """
        return BasisRegistry.get_basis(self)



#
def local_space(name, namespace = "", dimension = None, basis = None):
    """
    Create a LocalSpace with by default empty namespace string.
    If one also provides a set of basis states, these get stored via the BasisRegistry object.
    ALternatively, one may provide a dimension such that the states are simply labeled by a range of integers:
        [0, 1, 2, ..., dimension -1]

    :param name: Local space identifier
    :type name: str
    :param namespace: Local space namespace, see LocalSpace documentation
    :type namespace: str
    :param dimension: Dimension of local space (optional)
    :type dimension: int
    :param basis: Basis state labels for local space
    :type basis: sequence of int or str
    """
    s = LocalSpace.create(name, namespace)
    if dimension:
        if basis:
            assert len(basis) == dimension
        else:
            basis = xrange(dimension)
    if basis:
        BasisRegistry.set_basis(s, basis)
    return  s



def prod(sequence, neutral = 1):
    """
    Analog of the builtin `sum()` method.
    :param sequence: Sequence of objects that support being multiplied to each other.
    :type sequence: Any object that implements __mul__()
    :param neutral: The initial return value, which is also returned for zero-length sequence arguments.
    :type neutral: Any object that implements __mul__()
    :return: The product of the elements of `sequence`
    """
    return reduce(lambda a, b: a * b, sequence, neutral)


@flat
@idem
@check_signature_flat
@filter_neutral
class ProductSpace(HilbertSpace, Operation):
    """
    Tensor product space class for an arbitrary number of local space factors.
        ProductSpace(*factor_spaces)
    :param factor_spaces: The Hilbert spaces to be tensored together.
    :type factor_spaces: HilbertSpace

    """

    signature = HilbertSpace,
    neutral_element = TrivialSpace

    @classmethod
    def order_key(cls, a):
        if a is FullSpace:
            return (inf,)
        if a is TrivialSpace:
            return (0,)
        assert isinstance(a, LocalSpace)
        return a.operands

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
            oops = set((other,))
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
            other_ops = set((other,))
        return ProductSpace.create(*sorted(set(self.operands).intersection(other_ops)))

    def __lt__(self, other):
        if not isinstance(other, HilbertSpace):
            return NotImplemented
        if isinstance(other, ProductSpace):
            return set(self.operands) < set(other.operands)
        if other is FullSpace:
            return True
        return False

    def __gt__(self, other):
        if not isinstance(other, HilbertSpace):
            return NotImplemented
        if other is TrivialSpace:
            return True
        if isinstance(other, ProductSpace):
            return set(self.operands) > set(other.operands)
        if isinstance(other, LocalSpace):
            return other in self.operands
        return False

class BasisNotSetError(AlgebraError):
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
        :type basis: sequence of int or str
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

