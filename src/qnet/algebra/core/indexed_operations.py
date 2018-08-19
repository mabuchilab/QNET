"""Base classes for indexed operations (sums and products)"""

from abc import ABCMeta

from .abstract_algebra import Expression, Operation
from .exceptions import InfiniteSumError
from ..pattern_matching import wc
from ...utils.indices import (
    IdxSym, IndexRangeBase, SymbolicLabelBase, yield_from_ranges, )

__all__ = ["IndexedSum"]


class IndexedSum(Operation, metaclass=ABCMeta):
    """Base class for indexed sums"""

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
        """List of the dummy (index) variable symbols

        See also :property:`bound_symbols` for a set of the same symbols
        """
        return [r.index_symbol for r in self.ranges]

    @property
    def bound_symbols(self):
        """Set of bound variables, i.e. the index variable symbols

        See also :property:`variables` for an ordered list of the same symbols
        """
        return set(self.variables)

    @property
    def free_symbols(self):
        """Set of all free symbols"""
        return set([
            sym for sym in self.term.free_symbols
            if sym not in self.bound_symbols])

    @property
    def kwargs(self):
        return {}

    @property
    def terms(self):
        """Iterator over the terms of the sum

        Yield from the (possibly) infinite list of terms of the indexed sum, if
        the sum was written out explicitly. Each yielded term in an instance of
        :class:`.Expression`
        """
        from qnet.algebra.core.scalar_algebra import ScalarValue
        for mapping in yield_from_ranges(self.ranges):
            term = self.term.substitute(mapping)
            if isinstance(term, ScalarValue._val_types):
                term = ScalarValue.create(term)
            assert isinstance(term, Expression)
            yield term

    def __len__(self):
        length = 1
        for ind_range in self.ranges:
            try:
                length *= len(ind_range)
            except TypeError:
                raise InfiniteSumError(
                    "Cannot determine length from non-finite ranges")
        return length

    def doit(
            self, classes=None, recursive=True, indices=None, max_terms=None,
            **kwargs):
        """Write out the indexed sum explicitly

        If `classes` is None or :class:`IndexedSum` is in `classes`,
        (partially) write out the indexed sum in to an explicit sum of terms.
        If `recursive` is True, write out each of the new sum's summands by
        calling its :meth:`doit` method.

        Args:
            classes (None or list): see :meth:`.Expression.doit`
            recursive (bool): see :meth:`.Expression.doit`
            indices (list): List of :class:`IdxSym` indices for which the sum
                should be expanded. If `indices` is a subset of the indices
                over which the sum runs, it will be partially expanded. If not
                given, expand the sum completely
            max_terms (int): Number of terms after which to truncate the sum.
                This is particularly useful for infinite sums. If not given,
                expand all terms of the sum. Cannot be combined with `indices`
            kwargs: keyword arguments for recursive calls to
                :meth:`doit`. See :meth:`.Expression.doit`
        """
        return super().doit(
            classes, recursive, indices=indices, max_terms=max_terms, **kwargs)

    def _doit(self, **kwargs):
        indices = kwargs.get('indices', None)
        max_terms = kwargs.get('max_terms', None)
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
            return self
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
            res = res_term
        else:
            res = self.__class__.create(res_term, *other_ranges)
            res = res._doit_over_indices(indices=indices)
        return res

    def make_disjunct_indices(self, *others):
        """Return a copy with modified indices to ensure disjunct indices with
        `others`.

        Each element in `others` may be an index symbol (:class:`.IdxSym`),
        a index-range object (:class:`.IndexRangeBase`) or list of index-range
        objects, or an indexed operation (something with a ``ranges`` attribute)

        Each index symbol is primed until it does not match any index symbol in
        `others`.
        """
        new = self
        other_index_symbols = set()
        for other in others:
            try:
                if isinstance(other, IdxSym):
                    other_index_symbols.add(other)
                elif isinstance(other, IndexRangeBase):
                    other_index_symbols.add(other.index_symbol)
                elif hasattr(other, 'ranges'):
                    other_index_symbols.update(
                        [r.index_symbol for r in other.ranges])
                else:
                    other_index_symbols.update(
                        [r.index_symbol for r in other])
            except AttributeError:
                raise ValueError(
                    "Each element of other must be an an instance of IdxSym, "
                    "IndexRangeBase, an object with a `ranges` attribute "
                    "with a list of IndexRangeBase instances, or a list of"
                    "IndexRangeBase objects.")
        for r in self.ranges:
            index_symbol = r.index_symbol
            modified = False
            while index_symbol in other_index_symbols:
                modified = True
                index_symbol = index_symbol.incr_primed()
            if modified:
                new = new._substitute(
                    {r.index_symbol: index_symbol}, safe=True)
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
