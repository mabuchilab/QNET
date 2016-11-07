# coding=utf-8
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
Abstract Algebra
================

The abstract algebra package provides a basic interface
    for defining custom Algebras.

See :ref:`abstract_algebra` for more details.

"""
from __future__ import division

from abc import ABCMeta, abstractproperty
from functools import reduce
from sympy import (Basic as SympyBasic, Matrix as SympyMatrix)
from sympy.printing import latex as sympy_latex

from .pattern_matching import (
        ProtoExpr, match_pattern, wc, pattern_head, pattern)


def _trace(fn):
    """Function decorator to receive debugging information about function calls
    and return values.

    :param fn: Function whose calls to _trace
    :type fn: FunctionType
    :return: Decorated function
    :rtype: FunctionType
    """

    ### uncomment for debugging
    # def _tfn(*args, **kwargs):
    #     print("[", "-" * 40)
    #     ret = fn(*args, **kwargs)
    #     print("{}({},{}) called".format(fn.__name__, ", ".join(repr(a) for a in args),
    #                                     ", ".join(str(k) + "=" + repr(v) for k, v in kwargs.items())))
    #     print("-->", repr(ret))
    #     print("-" * 40, "]")
    #     return ret
    # return _tfn

    return fn


# define our own exceptions/errors
class AlgebraException(Exception):
    """Base class for all errors concerning the mathematical definitions and
    rules of an algebra."""
    pass


class AlgebraError(AlgebraException):
    """Base class for all errors concerning the mathematical definitions and
    rules of an algebra."""
    pass


class CannotSimplify(AlgebraException):
    """Raised when an expression cannot be further simplified"""
    pass


class WrongSignatureError(AlgebraError):
    """Raised when an operation is instantiated with operands of the wrong
    signature."""
    pass


class Expression(metaclass=ABCMeta):
    """Basic class defining the basic methods any Expression object should
    implement."""

    _rules = []
    _simplifications = []

    # hash, tex, and repr str, generated on demand (lazily)
    _hash = None
    _tex = None
    _repr = None

    @classmethod
    def create(cls, *args, **kwargs):
        """Instead of directly instantiating, it is recommended to use create,
        which applies simplifications to the args and keyword arguments
        according to the `_simplifications` class attribute, and returns an
        appropriate object (which may or may not be an instance of the original
        class)
        """
        for simplification in cls._simplifications:
            simplified = simplification(cls, args, kwargs)
            try:
                args, kwargs = simplified
            except (TypeError, ValueError):
                # We assume that if the simplification didn't return a tuple,
                # the result is a fully instantiated object
                return simplified
        if len(kwargs) > 0:
            cls._has_kwargs = True
        return cls(*args, **kwargs)

    @abstractproperty
    def args(self):
        """The tuple of positional arguments for the instantiation of the
        Expression"""
        raise NotImplementedError(self.__class__.__name__)

    @property
    def kwargs(self):
        """The dictionary of keyword arguments for the instantiation of the
        Expression"""
        if hasattr(self, '_has_kwargs') and self._has_kwargs:
            raise NotImplementedError(
                "Class %s does not provide a kwargs property"
                % str(self.__class__.__name__))
        return {}

    def __eq__(self, other):
        return (type(self) == type(other) and self.args == other.args and
                self.kwargs == other.kwargs)

    def __hash__(self):
        if self._hash is None:
            self._hash = expr_hash(self)
            if self._hash is None:
                raise TypeError("Cannot create hash")
        return self._hash

    def __repr__(self):
        if self._repr is None:
            args = self.args
            keys = sorted(self.kwargs.keys())
            kwargs_sep = ''
            if len(self.kwargs) > 0:
                kwargs_sep = ', '
            self._repr = (str(self.__class__.__name__) +
                          "(" +
                          ", ".join([repr(arg) for arg in args]) +
                          kwargs_sep +
                          ", ".join(["%s=%s" % (key, repr(self.kwargs[key]))
                                     for key in keys]) +
                          ")")
        return self._repr

    def __str__(self):
        return repr(self)

    def substitute(self, var_map):
        """Substitute all_symbols for other expressions.

        Args:
            var_map (dict): Dictionary with entries of the form ``{symbol:
                            substitution}``
        """
        return self._substitute(var_map)

    def _substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        return self.__class__.create(
                *map(lambda o: substitute(o, var_map), self.args),
                **self.kwargs)

    def tex(self):
        """Return a string containing a TeX-representation of self.
        Note that this needs to be wrapped by '$' characters for 'inline' LaTeX
        use.
        """
        if self._tex is None:
            args = self.args
            keys = sorted(self.kwargs.keys())
            self._repr = (r'{\rm %s}' % str(self.__class__.__name__) +
                          r'\left(' +
                          ", ".join([tex(arg) for arg in args]) +
                          ", ".join(["%s=%s" % (key, tex(self.kwargs[key]))
                                     for key in keys]) +
                          r'\right)')
        return self._tex

    def _repr_latex_(self):
        """For compatibility with the IPython notebook, generate TeX expression
        and surround it with $'s.
        """
        return "${}$".format(self.tex())

    def all_symbols(self):
        """Set of all_symbols contained within the expression."""
        return set_union(*[all_symbols(op) for op in self.args])

    def __ne__(self, other):
        """If it is well-defined (i.e. boolean), simply return
        the negation of ``self.__eq__(other)``
        Otherwise return NotImplemented.
        """
        eq = self.__eq__(other)
        if type(eq) is bool:
            return not eq
        return NotImplemented

    @classmethod
    def order_key(cls, obj):
        """Default ordering mechanism for achieving canonical ordering of
        expressions sequences."""
        try:
            return obj._order_key()
        except AttributeError:
            return str(obj)

    def _order_key(self):
        return (KeyTuple((self.__class__.__name__,) +
                tuple(map(Expression.order_key, self.args))))


def expr_hash(expr):
    """Return a hash for an expression"""
    h = hash((expr.__class__, tuple(expr.args), tuple(expr.kwargs.items())))
    return h


def substitute(expr, var_map):
    """(Safe) substitute, substitute objects for all symbols.

    Args:
        expr: The expression in which to perform the substitution
        var_map (dict): The substitution dictionary. See
            meth:`qnet.algebra.abstract_algebra.substitute` documentation
    """
    try:
        return expr.substitute(var_map)
    except AttributeError:
        if expr in var_map:
            return var_map[expr]
        return expr


def tex(obj):
    """Return a LaTeX string representation of the given `obj`"""
    if isinstance(obj, (int, float, complex)):
        return format_number_for_tex(obj)
    if isinstance(obj, (SympyBasic, SympyMatrix)):
        return sympy_latex(obj).strip('$')
    try:
        return obj.tex()
    except AttributeError:
        return r"{\rm " + str(obj) + "}"


def format_number_for_tex(num):
    """Format a floating or complex number of tex"""
    if num == 0:  #also True for 0., 0j
        return "0"
    if isinstance(num, complex):
        if num.real == 0:
            if num.imag == 1:
                return "i"
            if num.imag == -1:
                return "(-i)"
            if num.imag < 0:
                return "(-%si)" % format_number_for_tex(-num.imag)
            return "%si" % format_number_for_tex(num.imag)
        if num.imag == 0:
            return format_number_for_tex(num.real)
        return "(%s + %si)" % (format_number_for_tex(num.real),
                            format_number_for_tex(num.imag))
    if num < 0:
        return "(%g)" % num
    return "%g" % num


class KeyTuple(tuple):
    """A tuple that allows for ordering, facilitating the default ordering of
    Operations"""
    def __lt__(self, other):
        # print("<", self, other)
        if isinstance(other, (int, str)):
            return False
        if isinstance(other, KeyTuple):
            return super(KeyTuple, self).__lt__(other)
        raise AlgebraException("Cannot compare: {}".format(other))

    def __gt__(self, other):
        # print(">", self, other)
        if isinstance(other, (int, str)):
            return True
        if isinstance(other, KeyTuple):
            return super(KeyTuple, self).__gt__(other)
        raise AlgebraException("Cannot compare: {}".format(other))


def set_union(*sets):
    """Similar to ``sum()``, but for sets. Generate the union of an arbitrary
    number of set arguments.
    """
    # TODO: can this be done with set.union?
    return reduce(lambda a, b: a.union(b), sets, set(()))


def all_symbols(expr):
    """Return all all_symbols featured within an expression."""
    try:
        return expr.all_symbols()
    except AttributeError:
        return set(())


class Operation(Expression, metaclass=ABCMeta):
    """Abstract base class for all operations,
    where the operands themselves are also expressions.
    """

    def __init__(self, *operands, **kwargs):
        self._operands = operands
        if len(kwargs) > 0:
            raise ValueError("Unexpected keyword arguments: %s" %
                             (", ".join(sorted(kwargs.keys()))))
            # Any subclass of Operation that accepts keyword arguments must
            # define its own __init__

    @property
    def operands(self):
        """Tuple of operands of the operation"""
        return self._operands

    @property
    def args(self):
        """Alias for operands"""
        return self.operands

    def __getstate__(self):
        """state to be pickled"""
        d = self.__dict__.copy()
        if "_hash" in d:
            del d["_hash"]
        return d



###############################################################################
####################### ALGEBRAIC PROPERTIES FUNCTIONS ########################
###############################################################################


def assoc(cls, ops, kwargs):
    """Associatively expand out nested arguments of the flat class.
    E.g.::

        >>> class Plus(Operation):
        ...     _simplifications = [assoc, ]
        >>> Plus.create(1,Plus(2,3))
        Plus(1, 2, 3)
    """
    expanded = [(o,) if not isinstance(o, cls) else o.operands for o in ops]
    return sum(expanded, ()), kwargs


def idem(cls, ops, kwargs):
    """Remove duplicate arguments and order them via the cls's order_key key
    object/function.
    E.g.::

        >>> class Set(Operation):
        ...     _simplifications = [idem, ]
        >>> Set.create(1,2,3,1,3)
        Set(1, 2, 3)
    """
    return sorted(set(ops), key=cls.order_key), kwargs


def orderby(cls, ops, kwargs):
    """Re-order arguments via the class's ``order_key`` key object/function.
    Use this for commutative operations:
    E.g.::

        >>> class Times(Operation):
        ...     _simplifications = [orderby, ]
        >>> Times.create(2,1)
        Times(1, 2)
    """
    return sorted(ops, key=cls.order_key), kwargs


def filter_neutral(cls, ops, kwargs):
    """Remove occurrences of a neutral element from the argument/operand list,
    if that list has at least two elements.  To use this, one must also specify
    a neutral element, which can be anything that allows for an equality check
    with each argument.  E.g.::

        >>> class X(Operation):
        ...     neutral_element = 1
        ...     _simplifications = [filter_neutral, ]
        >>> X.create(2,1,3,1)
        X(2, 3)
    """
    c_n = cls.neutral_element
    if len(ops) == 0:
        return c_n
    fops = [op for op in ops if c_n != op]  # op != c_n does NOT work
    if len(fops) > 1:
        return fops, kwargs
    elif len(fops) == 1:
        # the remaining operand is the single non-trivial one
        return fops[0]
    else:
        # the original list of operands consists only of neutral elements
        return ops[0]


def match_replace(cls, ops, kwargs):
    """Match and replace a full operand specification to a function that
    provides a replacement for the whole expression
    or raises a :py:class:`CannotSimplify` exception.
    E.g.

    First define an operation::

        >>> class Invert(Operation):
        ...     _rules = []
        ...     _simplifications = [match_replace, ]

    Then some _rules::

        >>> A = wc("A")
        >>> A_float = wc("A", head=float)
        >>> Invert_A = pattern(Invert, A)
        >>> Invert._rules += [
        ...     (pattern_head(Invert_A), lambda A: A),
        ...     (pattern_head(A_float), lambda A: 1./A),
        ... ]

    Check rule application::

        >>> Invert.create("hallo")              # matches no rule
        Invert('hallo')
        >>> Invert.create(Invert("hallo"))      # matches first rule
        'hallo'
        >>> Invert.create(.2)                   # matches second rule
        5.0

    A pattern can also have the same wildcard appear twice::

        >>> class X(Operation):
        ...     _rules = [
        ...         (pattern_head(A, A), lambda A: A),
        ...     ]
        ...     _simplifications = [match_replace, ]
        >>> X.create(1,2)
        X(1, 2)
        >>> X.create(1,1)
        1

    """
    expr = ProtoExpr(ops, kwargs)
    for expr_or_pattern, replacement in cls._rules:
        match_dict = match_pattern(expr_or_pattern, expr)
        if match_dict:
            try:
                return replacement(**match_dict)
            except CannotSimplify:
                continue
    # No matching rules
    return ops, kwargs


def _get_binary_replacement(first, second, rules):
    """Helper function for match_replace_binary"""
    expr = ProtoExpr([first, second], {})
    for exp_or_pattern, replacement in rules:
        match_dict = match_pattern(exp_or_pattern, expr)
        if match_dict:
            try:
                return replacement(**match_dict)
            except CannotSimplify:
                continue
    return None


def match_replace_binary(cls, ops, kwargs):
    """Similar to func:`match_replace`, but for arbitrary length operations,
    such that each two pairs of subsequent operands are matched pairwise.

        >>> A = wc("A")
        >>> class FilterDupes(Operation):
        ...     _binary_rules = [(pattern_head(A,A), lambda A: A), ]
        ...     _simplifications = [match_replace_binary, ]
        >>> FilterDupes.create(1,2,3,4)         # No duplicates
        FilterDupes(1, 2, 3, 4)
        >>> FilterDupes.create(1,2,2,3,4)       # Some duplicates
        FilterDupes(1, 2, 3, 4)

    Note that this only works for *subsequent* duplicate entries:

        >>> FilterDupes.create(1,2,3,2,4)       # No *subsequent* duplicates
        FilterDupes(1, 2, 3, 2, 4)
    """
    ops = list(ops)
    rules = cls._binary_rules
    j = 1
    while j < len(ops):
        first, second = ops[j - 1], ops[j]
        replacement = _get_binary_replacement(first, second, rules)
        if replacement is not None:
            if assoc in cls._simplifications and isinstance(replacement, cls):
                ops = ops[:j - 1] + list(replacement.operands) + ops[j + 1:]
            else:
                ops = ops[:j - 1] + [replacement,] + ops[j + 1:]
            if j > 1:
                j -= 1
        else:
            j += 1
    return ops, kwargs


# Store all singletons in a dict
_singletons = {}
def _get_singleton(name):
    """Retrieve singletons by name."""
    return _singletons[name]

def singleton(cls):
    """Singleton class decorator. Turns a class object into a unique instance.

    :param cls: Class to decorate
    :type cls: type
    :return: The singleton instance of that class
    :rtype: cls
    """

    # noinspection PyDocstring
    class S(cls):
        __instance = None

        def __hash__(self):
            return hash(cls)

        # noinspection PyMethodMayBeStatic
        def _symbols(self):
            return set(())

        def __repr__(self):
            return cls.__name__

        def __call__(self):
            return self.__instance

        def __reduce__(self):
            """
            This magic method ensures that singletons can be pickled.
            See also https://docs.python.org/3.1/library/pickle.html#pickle.object.__reduce__
            """
            return (_get_singleton, (cls.__name__,))

    S.__name__ = cls.__name__
    S.__instance = s = S()
    _singletons[S.__name__] = s

    return s
