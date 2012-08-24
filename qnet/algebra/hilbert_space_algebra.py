#!/usr/bin/env python
# encoding: utf-8

from qnet.algebra.abstract_algebra import *


#class SpaceExists(AlgebraError):
#    pass
#


class HilbertSpace(object):

    def tensor(self, other):
        return ProductSpace.create(self, other)

    def remove(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def intersect(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def local_factors(self):
        raise NotImplementedError(self.__class__.__name__)

    def __len__(self):
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

    @property
    def dimension(self):
        raise NotImplementedError(self.__class__.__name__)








def singleton(cls):
    s = cls()
    s.__call__ = lambda : s
    return s

@singleton
class TrivialSpace(HilbertSpace):
    def tensor(self, other):
        return other

    def remove(self, other):
        return self

    def intersect(self, other):
        return self

    def __hash__(self):
        return hash(self.__class__)

    def local_factors(self):
        return ()

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return self is other

    dimension = 0


@singleton
class FullSpace(HilbertSpace):
    def tensor(self, other):
        return self

    def remove(self, other):
        raise AlgebraError()

    def local_factors(self):
        raise AlgebraError()


    def intersect(self, other):
        return other

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return self is other



    dimension = NotImplemented



@singleton
class InfiniteNumbers(object):
    def __iter__(self):
        def infinite_numbers():
            i = 0
            while True:
                yield i
                i += 1
        return infinite_numbers()

    def index(self, state):
        if isinstance(state, int):
            return state
        raise ValueError()


class Basis(object):
    dimension = inf
    states = InfiniteNumbers

    def __init__(self, states = None):
        if states:
            self.states = states
            self.dimension = len(states)

    def __repr__(self):
        if isinstance(self.states, tuple):
            return "Basis({})".format(repr(self.states))
        if self.dimension < inf:
            return "Basis(xrange({}))".format(str(self.dimension))
        return "Basis()"

    def __eq__(self, other):
        return other.__class__ == Basis and self.states == other.states and self.dimension == other.dimension





def local_space(name, namespace = "", dimension = None, basis = None):
    if basis:
        if not isinstance(basis, Basis):
            basis = Basis(basis)
        if dimension:
            assert basis.dimension == dimension
        return LocalSpace.create(name, namespace, basis)
    elif dimension:
        basis = Basis(xrange(dimension))
    else:
        basis = Basis()
    return LocalSpace.create(name, namespace, basis)



@check_signature
class LocalSpace(HilbertSpace, Operation):
    signature = str, str, Basis

    def __eq__(self, other):
        "Return True if name and namespace match. Local spaces are uniquely determined by name and namespace."
        return type(self) == type(other) and self.operands[:2] == other.operands[:2]

    def __hash__(self):
        if not self._hash:
            self._hash = hash((self.__class__, self.operands[:2]))
        return self._hash



    def remove(self, other):
        if other == self:
            return TrivialSpace
        return self

    def intersect(self, other):
        if other == self:
            return self
        return TrivialSpace

    def local_factors(self):
        return (self,)

    @property
    def dimension(self):
        return self.basis.dimension

    @property
    def basis(self):
        return self.operands[2]

    @property
    def states(self):
        return self.basis.states

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


def prod(sequence, neutral = 1):
    return reduce(lambda a, b: a * b, sequence, neutral)


@flat
@idem
@check_signature_flat
@filter_neutral
class ProductSpace(HilbertSpace, Operation):
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

    def remove(self, other):
        if other is FullSpace:
            return TrivialSpace
        if other is TrivialSpace:
            return self
        if isinstance(other, ProductSpace):
            oops = set(other.operands)
        else:
            oops = set((other,))
        return ProductSpace.create(*sorted(set(self.operands).difference(oops)))

    def local_factors(self):
        return self.operands

    def intersect(self, other):
        if other is FullSpace:
            return self
        if other is TrivialSpace:
            return TrivialSpace
        if isinstance(other, ProductSpace):
            other_ops = set(other.operands)
        else:
            other_ops = set((other,))
        return ProductSpace.create(*sorted(set(self.operands).intersection(other_ops)))

    @property
    def dimension(self):
        return prod((o.dimension for o in self.operands), neutral = TrivialSpace)

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

