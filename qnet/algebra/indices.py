from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from itertools import product
import attr
import sympy

from ..printing import srepr, ascii

__all__ = [
    'IntIndex', 'FockIndex', 'StrLabel', 'IndexOverList', 'IndexOverRange',
    'IndexOverFockSpace', 'KroneckerDelta']

__private__ = [
    'yield_from_ranges', 'SymbolicLabelBase', 'IndexRangeBase']


# support routines


def KroneckerDelta(i, j):
    """Kronecker delta function.

    If ``i == j``, return 1. Otherwise,
    If ``i != j``, if `i` and `j` are Sympy or SymbolicLabelBase objects,
    return an instance of :class:`sympy.KroneckerDelta`, return 0 otherwise.

    Unlike in :class:`sympy.KroneckerDelta`, `i` and `j` will not be sympyfied
    """
    if i == j:
        return 1
    else:
        if isinstance(i, sympy.Basic) and isinstance(j, sympy.Basic):
            return sympy.KroneckerDelta(i, j)
        elif (
                isinstance(i, SymbolicLabelBase) and
                isinstance(j, SymbolicLabelBase)):
            return sympy.KroneckerDelta(i.expr, j.expr)

        else:
            return 0


def _immutable_attribs(cls):
    """Class decorator like ``attr.s(frozen=True)`` with improved __repr__"""
    cls = attr.s(cls, frozen=True)
    defaults = OrderedDict([(a.name, a.default) for a in cls.__attrs_attrs__])

    def repr_(self):
        real_cls = self.__class__
        class_name = real_cls.__name__
        args = []
        for name in defaults.keys():
            val = getattr(self, name)
            positional = defaults[name] == attr.NOTHING
            if val != defaults[name]:
                args.append(
                    srepr(val) if positional else "%s=%s" % (name, srepr(val)))
        return "{0}({1})".format(class_name, ", ".join(args))

    cls.__repr__ = repr_
    return cls


def _merge_dicts(*dicts):
    """Given any number of dicts, shallow copy and merge into a new dict."""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def yield_from_ranges(ranges):
    range_iters = []
    for index_range in ranges:
        range_iters.append(index_range.iter)
    for dicts in product(*range_iters):
        yield _merge_dicts(*dicts)


# Classes for symbolic labels


@_immutable_attribs
class SymbolicLabelBase(metaclass=ABCMeta):
    expr = attr.ib(validator=attr.validators.instance_of(sympy.Basic))

    @abstractmethod
    def evaluate(self, mapping):
        pass

    def _sympy_(self):
        # sympyfication allows the symbolic label to be used in other sympy
        # expressions (which happens in some algebraic rules)
        return self.expr


class IntIndex(SymbolicLabelBase):

    def evaluate(self, mapping):
        return int(self.expr.subs(mapping))


class FockIndex(IntIndex):
    pass


class StrLabel(SymbolicLabelBase):

    def evaluate(self, mapping):
        return ascii(self.expr.subs(mapping))


# Index Ranges


@_immutable_attribs
class IndexRangeBase(metaclass=ABCMeta):
    index_symbol = attr.ib(validator=attr.validators.instance_of(sympy.Basic))

    @abstractproperty
    def iter(self):
        pass


@_immutable_attribs
class IndexOverList(IndexRangeBase):
    values = attr.ib(convert=tuple)

    @property
    def iter(self):
        for val in self.values:
            yield {self.index_symbol: val}


@_immutable_attribs
class IndexOverRange(IndexRangeBase):
    start_from = attr.ib(validator=attr.validators.instance_of(int))
    to = attr.ib(validator=attr.validators.instance_of(int))
    step = attr.ib(validator=attr.validators.instance_of(int), default=1)

    @property
    def iter(self):
        ind_range = range(
            self.start_from,
            (self.to + 1) if self.step >= 0 else (self.to - 1),
            self.step)
        for ind in ind_range:
            yield {self.index_symbol: ind}


@_immutable_attribs
class IndexOverFockSpace(IndexRangeBase):
    hs = attr.ib()
    # TODO: assert that hs is indeed a FockSpace

    @property
    def iter(self):
        if self.hs._dimension is None:
            i = 0
            while True:
                yield {self.index_symbol: i}
                i += 1
        else:
            for ind in range(self.hs.dimension):
                yield {self.index_symbol: ind}
