"""Base classes for indexed operations (sums and products)"""

from abc import ABCMeta

import attr
from sympy import KroneckerDelta as SympyKroneckerDelta, Mul as SympyMul

from .abstract_algebra import Operation, ScalarTimesExpression, all_symbols
from .exceptions import InfiniteSumError
from ..pattern_matching import wc
from ...utils.indices import (
    IdxSym, IndexRangeBase, SymbolicLabelBase, yield_from_ranges, )
from ...utils.ordering import KeyTuple, expr_order_key

__all__ = [
    "IndexedSum", "indexed_sum_over_const", "indexed_sum_over_kronecker"]


class IndexedSum(Operation, metaclass=ABCMeta):
    # TODO: documentation

    _expanded_cls = None  # must be set by subclasses
    _expand_limit = 1000

    def __init__(self, term, *ranges):
        self._term = term
        self.ranges = tuple(ranges)
        for r in self.ranges:
            if not isinstance(r, IndexRangeBase):
                # We need this type check to that we can use attr.astuple below
                raise TypeError(
                    "Every range must be an instance of IndexRangeBase")

        self._order_key = KeyTuple((
            self.__class__.__name__, '__', 1.0, expr_order_key(term),
            tuple([attr.astuple(r) for r in ranges])))

        index_symbols = set([r.index_symbol for r in ranges])
        if len(index_symbols) != len(self.ranges):
            raise ValueError(
                "ranges %s must have distinct index_symbols" % repr(ranges))
        super().__init__(term, ranges=ranges)

    @property
    def term(self):
        return self._term

    @property
    def operands(self):
        return (self._term, )

    @property
    def args(self):
        return tuple([self._term, *self.ranges])

    @property
    def variables(self):
        """List of the dummy (index) variable symbols"""
        return [r.index_symbol for r in self.ranges]

    def all_symbols(self):
        """Set of all free symbols"""
        return set(
            [sym for sym in all_symbols(self.term)
                if sym not in self.variables])

    @property
    def kwargs(self):
        return {}

    @property
    def terms(self):
        for mapping in yield_from_ranges(self.ranges):
            yield self.term.substitute(mapping).simplify(rules=[(
                wc('label', head=SymbolicLabelBase),
                lambda label: label.evaluate(mapping))])

    def __len__(self):
        length = 1
        for ind_range in self.ranges:
            try:
                length *= len(ind_range)
            except TypeError:
                raise InfiniteSumError(
                    "Cannot determine length from non-finite ranges")
        return length

    def doit(self, indices=None, max_terms=None):
        if indices is None:
            return self._doit_full(max_terms=max_terms)
        else:
            if max_terms is not None:
                raise ValueError(
                    "max_terms is incompatible with summing over specific "
                    "indices")
            return self._doit_over_indices(indices)

    def _doit_full(self, max_terms=None):
        res = None
        if max_terms is None:
            len(self)  # side-effect: raise InfiniteSumError
        else:
            if max_terms > self._expand_limit:
                raise ValueError(
                    "max_terms = %s must be smaller than the limit %s"
                    % (max_terms, self._expand_limit))
        for i, term in enumerate(self.terms):
            if max_terms is not None:
                if i >= max_terms:
                    break
            if res is None:
                res = term
            else:
                res += term
            if i > self._expand_limit:
                raise InfiniteSumError(
                    "Cannot expand %s: more than %s terms"
                    % (self, self._expand_limit))
        return res

    def _doit_over_indices(self, indices):
        if len(indices) == 0:
            return self
        ind_sym, *indices = indices
        if not isinstance(ind_sym, IdxSym):
            ind_sym = IdxSym(ind_sym)
        selected_range = None
        other_ranges = []
        for index_range in self.ranges:
            if index_range.index_symbol == ind_sym:
                selected_range = index_range
            else:
                other_ranges.append(index_range)
        if selected_range is None:
            raise ValueError(
                "Index %s does not appear in %s" % (ind_sym, self))
        res_term = None
        for i, mapping in enumerate(selected_range.iter()):
            res_summand = self.term.substitute(mapping)
            if res_term is None:
                res_term = res_summand
            else:
                res_term += res_summand
            if i > self._expand_limit:
                raise InfiniteSumError(
                    "Cannot expand %s: more than %s terms"
                    % (self, self._expand_limit))
        if len(other_ranges) == 0:
            res = res_term.simplify(rules=[(
                wc('label', head=SymbolicLabelBase),
                lambda label: label.evaluate(mapping))])
        else:
            res = self.__class__.create(res_term, *other_ranges)
            res = res._doit_over_indices(indices=indices)
        return res

    def make_disjunct_indices(self, *others):
        """Return a copy with modified indices to ensure disjunct indices with
        `others`.

        Each index symbol is primed until it does not match any index symbol in
        `others`
        """
        new = self
        other_index_symbols = set()
        for other in others:
            try:
                if isinstance(other, IndexRangeBase):
                    other_index_symbols.add(other.index_symbol)
                elif hasattr(other, 'ranges'):
                    other_index_symbols.update(
                        [r.index_symbol for r in other.ranges])
                else:
                    other_index_symbols.update(
                        [r.index_symbol for r in other])
            except AttributeError:
                raise ValueError(
                    "Each element of other must be an an instance of "
                    "IndexRangeBase, and object with a `ranges` attribute "
                    "with a list of IndexRangeBase instances, or a list of"
                    "IndexRangeBase objects directly")
        for r in self.ranges:
            index_symbol = r.index_symbol
            while index_symbol in other_index_symbols:
                index_symbol = index_symbol.incr_primed()
            new = new._substitute({r.index_symbol: index_symbol}, safe=True)
        return new

    def __mul__(self, other):
        if isinstance(other, IndexedSum):
            other = other.make_disjunct_indices(self)
            new_ranges = self.ranges + other.ranges
            return self.__class__.create(self.term * other.term, *new_ranges)
        try:
            return super().__mul__(other)
        except AttributeError:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, IndexedSum):
            self = self.make_disjunct_indices(other)
            new_ranges = other.ranges + self.ranges
            return self.__class__.create(other.term * self.term, *new_ranges)
        try:
            return super().__rmul__(other)
        except AttributeError:
            return NotImplemented

    def __add__(self, other):
        raise NotImplementedError()
        if isinstance(other, self.__class__):
            # TODO: this is wrong!!!!!!
            if set(self.variables).isdisjoint(other.variables):
                new_ranges = self.ranges + other.ranges
                return self.__class__.create(
                    self.term + other.term, *new_ranges)
        try:
            return super().__add__(other)
        except AttributeError:
            return NotImplemented

    def __radd__(self, other):
        raise NotImplementedError()
        if isinstance(other, self.__class__):
            # TODO: this is wrong!!!!!!
            if set(self.variables).isdisjoint(other.variables):
                new_ranges = other.ranges + self.ranges
                return self.__class__.create(
                    other.term + self.term, *new_ranges)
        try:
            return super().__radd__(other)
        except AttributeError:
            return NotImplemented

    def __sub__(self, other):
        raise NotImplementedError()
        if isinstance(other, self.__class__):
            # TODO: this is wrong!!!!!!
            if set(self.variables).isdisjoint(other.variables):
                new_ranges = self.ranges + other.ranges
                return self.__class__.create(
                    self.term - other.term, *new_ranges)
        try:
            return super().__sub__(other)
        except AttributeError:
            return NotImplemented

    def __rsub__(self, other):
        raise NotImplementedError()
        if isinstance(other, self.__class__):
            # TODO: this is wrong!!!!!!
            if set(self.variables).isdisjoint(other.variables):
                new_ranges = other.ranges + self.ranges
                return self.__class__.create(
                    other.term - self.term, *new_ranges)
        try:
            return super().__rsub__(other)
        except AttributeError:
            return NotImplemented


def indexed_sum_over_const(cls, ops, kwargs):
    """Execute an indexed sum over a term that does not depend on the summation
    indices

    ..math::

        \sum_{j=1}{N} a = N a
    """
    term, *ranges = ops
    bound_symbols = set([r.index_symbol for r in ranges])
    if len(all_symbols(term).intersection(bound_symbols)) == 0:
        n = 1
        for r in ranges:
            try:
                n *= len(r)
            except TypeError:
                return ops, kwargs
        return n * term
    else:
        return ops, kwargs


def indexed_sum_over_kronecker(cls, ops, kwargs):
    """Execute sums over KroneckerDelta prefactors"""
    term, *ranges = ops
    correct_structure = (
        isinstance(term, ScalarTimesExpression) and
        isinstance(term.coeff, SympyMul) and
        len(ranges) >= 2)
    if correct_structure:
        bound_symbols = set([r.index_symbol for r in ranges])
        for factor in term.coeff.args:
            if isinstance(factor, SympyKroneckerDelta):
                i, j = factor.args
                assert i in bound_symbols and j in bound_symbols
                if i.primed > j.primed:
                    # we prefer eliminating indices with more prime dashes
                    i, j = j, i
                term = term.substitute({factor: 1, j: i})
                ranges = [r for r in ranges if r.index_symbol != j]
        ops = (term, ) + tuple(ranges)
    return ops, kwargs
