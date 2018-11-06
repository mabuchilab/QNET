import re
from abc import ABCMeta, abstractmethod

import attr
import sympy
from sympy import Piecewise
from sympy.core.cache import cacheit as sympy_cacheit

from ._attrs import immutable_attribs

__all__ = [
    'IdxSym', 'IntIndex', 'FockIndex', 'StrLabel', 'FockLabel', 'SpinIndex',
    'IndexOverList', 'IndexOverRange', 'IndexOverFockSpace']

__private__ = [
    'yield_from_ranges', 'SymbolicLabelBase', 'IndexRangeBase', 'product']


# support routines


def _merge_dicts(*dicts):
    """Given any number of dicts, shallow copy and merge into a new dict"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def product(*generators, repeat=1):
    """Cartesian product akin to :func:`itertools.product`, but accepting
    generator functions

    Unlike :func:`itertools.product` this function does not convert the input
    iterables into tuples. Thus, it can handle large or infinite inputs. As a
    drawback, however, it only works with "restartable" iterables (something
    that :func:`iter` can repeatably turn into an iterator, or a generator
    function (but not the generator iterator that is returned by that
    generator function)

    Args:
        generators: list of restartable iterators or generator functions
        repeat: number of times `generators` should be repeated

    Adapted from https://stackoverflow.com/q/12093364/
    """
    if len(generators) == 0:
        yield ()
    else:
        generators = generators * repeat
        it = generators[0]
        for item in it() if callable(it) else iter(it):
            for items in product(*generators[1:]):
                yield (item, ) + items


def yield_from_ranges(ranges):
    range_iters = []
    for index_range in ranges:
        assert callable(index_range.iter)
        # index_range.iter must be a generator (so that it's restartable),
        # *not* an iterator, which would be index_range.iter()
        range_iters.append(index_range.iter)
    for dicts in product(*range_iters):
        yield _merge_dicts(*dicts)


class IdxSym(sympy.Symbol):
    """Index symbol in an indexed sum or product

    Args:
        name (str): The label for the symbol. It must be a simple Latin or
            Greek letter, possibly with a subscript, e.g. ``'i'``, ``'mu'``,
            ``'gamma_A'``
        primed (int): Number of prime marks (') associated with the symbol

    Notes:

        The symbol can be used in arbitrary algebraic (sympy) expressions::

            >>> sympy.sqrt(IdxSym('n') + 1)
            sqrt(n + 1)

        By default, the symbol is assumed to represent an integer. If this is
        not the case, you can instantiate explicitly as a non-integer::

            >>> IdxSym('i').is_integer
            True
            >>> IdxSym('i', integer=False).is_integer
            False

        You may also declare the symbol as positive::

            >>> IdxSym('i').is_positive
            >>> IdxSym('i', positive=True).is_positive
            True

        The `primed` parameter is used to automatically create distinguishable
        indices in products of sums, or more generally if the same index occurs
        in an expression with potentially differnt values::

            >>> ascii(IdxSym('i', primed=2))
            "i''"
            >>> IdxSym('i') == IdxSym('i', primed=1)
            False

        It should not be used when creating indices "by hand"

    Raises:
        ValueError: if `name` is not a simple symbol label, or if primed < 0
        TypeError: if `name` is not a string
    """

    is_finite = True
    is_Symbol = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True

    _rx_name = re.compile('^[A-Za-z]+(_[A-Za-z0-9().,+-]+)?$')

    def __new_stage2__(cls, name, primed=0, **kwargs):
        # remove: start, stop, points
        if not cls._rx_name.match(name):
            raise ValueError(
                "name '%s' does not match pattern '%s'"
                % (name, cls._rx_name.pattern))
        primed = int(primed)
        if not primed >= 0:
            raise ValueError("`primed` must be an integer >= 0")
        if 'integer' not in kwargs:
            kwargs['integer'] = True
        obj = super().__xnew__(cls, name, **kwargs)
        obj.params = (primed, )
        obj._primed = primed
        return obj

    def __new__(cls, name, *, primed=0, **kwargs):
        obj = cls.__xnew_cached_(cls, name, primed, **kwargs)
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sympy_cacheit(__new_stage2__))

    def _hashable_content(self):
        return (sympy.Symbol._hashable_content(self), self.params)

    @property
    def primed(self):
        return self._primed

    def incr_primed(self, incr=1):
        """Return a copy of the index with an incremented :attr:`primed`"""
        return self.__class__(
            self.name, primed=self._primed + incr,
            **self._assumptions.generator)

    @property
    def prime(self):
        """equivalent to :meth:`inc_primed` with ``incr=1``"""
        return self.incr_primed(incr=1)

    def _sympystr(self, printer, *args):
        return printer._print_Symbol(self) + "'" * self.primed

    def _sympyrepr(self, printer, *args):
        res = printer._print_Symbol(self)
        if self.primed > 0:
            res = res[:-1] + ", primed=%d)" % self.primed
        return res

    def _pretty(self, printer, *args):
        res = printer._print_Symbol(self)
        return res.__class__(
            res.s + "'" * self.primed, res.baseline, res.binding)

    def _latex(self, printer, *args):
        res = printer._print_Symbol(self)
        if self.primed > 0:
            res = r'{%s^{%s}}' % (res, r'\prime' * self.primed)
        return res


# Classes for symbolic labels


@immutable_attribs
class SymbolicLabelBase(metaclass=ABCMeta):
    """Base class for symbolic labels

    A symbolic label is a SymPy expression that contains one or more
    :class:`IdxSym`, and can be rendered into an integer or string label by
    substituting integer values for each :class:`IdxSym`.

    See :class:`IntIndex` for an example.
    """
    expr = attr.ib()

    @expr.validator
    def _validate_expr(self, attribute, value):
        if not isinstance(value, sympy.Basic):
            raise ValueError("expr must be a sympy formula")
        if not self._has_idx_syms(value):
            raise ValueError("expr must contain at least one IdxSym")

    @staticmethod
    def _has_idx_syms(expr):
        return any([isinstance(sym, IdxSym) for sym in expr.free_symbols])

    @abstractmethod
    def _render(self, expr):
        """Render `expr` into a a label. It can be assumed that `expr` does not
        contain any :class:`IdxSym`"""
        pass

    def substitute(self, var_map):
        """Substitute in the expression describing the label.

        If the result of the substitution no longer contains any
        :class:`IdxSym`, this returns a "rendered" label.
        """
        new_expr = self.expr.subs(var_map)
        if self._has_idx_syms(new_expr):
            return self.__class__(expr=new_expr)
        else:
            return self._render(new_expr)

    @property
    def free_symbols(self):
        """Free symbols in the expression describing the label"""
        return self.expr.free_symbols


class IntIndex(SymbolicLabelBase):
    """A symbolic label that evaluates to an integer

    The label can be rendered via :meth:`substitute`::

        >>> i, j = symbols('i, j', cls=IdxSym)
        >>> idx = IntIndex(i+j)
        >>> idx.substitute({i: 1, j:1})
        2

    An "incomplete" substitution (anything that still leaves a :class:`IdxSym`
    in the label expression) will result in another :class:`IntIndex`
    instance::

        >>> idx.substitute({i: 1})
        IntIndex(Add(IdxSym('j', integer=True), Integer(1)))
    """

    def __mul__(self, other):
        return other * self.expr

    def __add__(self, other):
        return self.expr + other

    def _render(self, expr):
        return int(expr)


class FockIndex(IntIndex):
    """Symbolic index labeling a basis state in a :class:`.LocalSpace`"""

    @property
    def fock_index(self):
        return self.expr


class StrLabel(SymbolicLabelBase):
    """Symbolic label that evaluates to a string

    Example:
        >>> i = symbols('i', cls=IdxSym)
        >>> A = symbols('A', cls=sympy.IndexedBase)
        >>> lbl = StrLabel(A[i])
        >>> lbl.substitute({i: 1})
        'A_1'
    """

    def _render(self, expr):
        from qnet.printing.sympy import SympyStrPrinter
        return SympyStrPrinter().doprint(expr)


@immutable_attribs
class FockLabel(StrLabel):
    """Symbolic label that evaluates to the label of a basis state

    This evaluates first to an index, and then to the label for the basis state
    of the Hilbert space for that index::

        >>> hs = LocalSpace('tls', basis=('g', 'e'))
        >>> i = symbols('i', cls=IdxSym)
        >>> lbl = FockLabel(i, hs=hs)
        >>> lbl.substitute({i: 0})
        'g'
    """
    hs = attr.ib()

    def _render(self, expr):
        i = int(expr)
        return self.hs.basis_labels[i]

    @property
    def fock_index(self):
        return self.expr

    def substitute(self, var_map):
        """Substitute in the expression describing the label.

        If the result of the substitution no longer contains any
        :class:`IdxSym`, this returns a "rendered" label.
        """
        new_expr = self.expr.subs(var_map)
        new_hs = self.hs.substitute(var_map)
        if self._has_idx_syms(new_expr):
            return self.__class__(expr=new_expr, hs=new_hs)
        else:
            return self._render(new_expr)


@immutable_attribs
class SpinIndex(StrLabel):
    """Symbolic label for a spin degree of freedom

    This evaluates to a string representation of an integer or half-integer.
    For values of e.g.  1, -1, 1/2, -1/2, the rendered resulting string is
    "+1", "-1", "+1/2", "-1/2", respectively (in agreement with the convention
    for the basis labels in a spin degree of freedom)

        >>> i = symbols('i', cls=IdxSym)
        >>> hs = SpinSpace('s', spin='1/2')
        >>> lbl = SpinIndex(i/2, hs)
        >>> lbl.substitute({i: 1})
        '+1/2'

    Rendering an expression that is not integer or half-integer valued results
    in a :exc:`ValueError`.
    """
    hs = attr.ib()

    @hs.validator
    def _validate_hs(self, attribute, value):
        from qnet.algebra.library.spin_algebra import SpinSpace
        if not isinstance(value, SpinSpace):
            raise ValueError("hs must be a SpinSpace instance")

    def _render(self, expr):
        return self._static_render(expr)

    @staticmethod
    def _static_render(expr):
        if expr.is_integer:
            int_val = int(expr)
            if int_val > 0:
                return "+" + str(int_val)
            else:
                return str(int_val)
        else:  # half-integer
            numer, denom = expr.as_numer_denom()
            if not (numer.is_integer and denom == 2):
                raise ValueError(
                    "SpinIndex must evaluate to an integer or "
                    "half-integer, not %s" % str(expr))
            if numer > 0:
                return "+%d/%d" % (int(numer), int(denom))
            else:
                return "%d/%d" % (int(numer), int(denom))

    @property
    def fock_index(self):
        return self.expr + self.hs.spin

    def substitute(self, var_map):
        """Substitute in the expression describing the label.

        If the result of the substitution no longer contains any
        :class:`IdxSym`, this returns a "rendered" label.
        """
        new_expr = self.expr.subs(var_map)
        new_hs = self.hs.substitute(var_map)
        if self._has_idx_syms(new_expr):
            return self.__class__(expr=new_expr, hs=new_hs)
        else:
            return self._render(new_expr)


# Index Ranges


@immutable_attribs
class IndexRangeBase(metaclass=ABCMeta):
    """Base class for index ranges

    Index ranges occur in indexed sums or products.
    """
    index_symbol = attr.ib(validator=attr.validators.instance_of(IdxSym))

    @abstractmethod
    def iter(self):
        # this should *not* be a property: for `product`, we need to pass a
        # generator function, i.e. IndexRangeBase.iter, not
        # IndexRangeBase.iter()
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __contains__(self, val):
        raise NotImplementedError()

    @abstractmethod
    def substitute(self, var_map):
        raise NotImplementedError()

    @abstractmethod
    def piecewise_one(self, expr):
        """Value of 1 for all index values in the range, 0 otherwise

        A :class:`~sympy.functions.elementary.piecewise.Piecewise` object that
        is 1 for any value of `expr` in the range of possible index values,
        and 0 otherwise.
        """
        raise NotImplementedError()


@immutable_attribs
class IndexOverList(IndexRangeBase):
    """Index over a list of explicit values

    Args:
        index_symbol (IdxSym): The symbol iterating over the value
        values (list): List of values for the index
    """
    values = attr.ib(convert=tuple)

    def iter(self):
        for val in self.values:
            yield {self.index_symbol: val}

    def __len__(self):
        return len(self.values)

    def __contains__(self, val):
        return val in self.values

    def substitute(self, var_map):
        new_index_symbol = var_map.get(self.index_symbol, self.index_symbol)
        new_values = tuple(
            [var_map.get(element, element) for element in self.values])
        return self.__class__(index_symbol=new_index_symbol, values=new_values)

    def piecewise_one(self, expr):
        return Piecewise(
            (1, sympy.FiniteSet(*self.values).contains(expr)),
            (0, True))


@immutable_attribs
class IndexOverRange(IndexRangeBase):
    """Index over the inclusive range between two integers

    Args:
        index_symbol (IdxSym): The symbol iterating over the range
        start_from (int): Starting value for the index
        to (int): End value of the index
        step (int): Step width by which index increases
    """
    start_from = attr.ib(validator=attr.validators.instance_of(int))
    to = attr.ib(validator=attr.validators.instance_of(int))
    step = attr.ib(validator=attr.validators.instance_of(int), default=1)

    def iter(self):
        for ind in self.range:
            yield {self.index_symbol: ind}

    @property
    def range(self):
        return range(
            self.start_from,
            (self.to + 1) if self.step >= 0 else (self.to - 1),
            self.step)

    def __len__(self):
        return len(self.range)

    def __contains__(self, val):
        return val in self.range

    def substitute(self, var_map):
        new_index_symbol = var_map.get(self.index_symbol, self.index_symbol)
        return self.__class__(
            index_symbol=new_index_symbol, start_from=self.start_from,
            to=self.to, step=self.step)

    def piecewise_one(self, expr):
        return Piecewise(
            (1, sympy.FiniteSet(*list(self.range)).contains(expr)),
            (0, True))


@immutable_attribs
class IndexOverFockSpace(IndexRangeBase):
    """Index range over the integer indices of a :class:`.LocalSpace` basis

    Args:
        index_symbol (IdxSym): The symbol iterating over the range
        hs (.LocalSpace): Hilbert space over whose basis to iterate

    """
    hs = attr.ib()
    # TODO: assert that hs is indeed a LocalSpace

    def iter(self):
        if self.hs._dimension is None:
            i = 0
            while True:
                yield {self.index_symbol: i}
                i += 1
        else:
            for ind in range(self.hs.dimension):
                yield {self.index_symbol: ind}

    def __len__(self):
        return self.hs._dimension

    def __contains__(self, val):
        if self.hs._dimension is None:
            return val >= 0
        else:
            return 0 <= val < self.hs.dimension

    def substitute(self, var_map):
        new_index_symbol = var_map.get(self.index_symbol, self.index_symbol)
        new_hs = var_map.get(self.hs, self.hs)
        return self.__class__(index_symbol=new_index_symbol, hs=new_hs)

    def piecewise_one(self, expr):
        if self.hs._dimension is None:
            to_val = sympy.oo
        else:
            to_val = self.hs.dimension
        return Piecewise(
            (1, sympy.Interval(0, to_val).as_relational(expr)),
            (0, True))
