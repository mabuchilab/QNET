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
    """
    def _tfn(*args, **kwargs):
        print("[", "-" * 40)
        ret = fn(*args, **kwargs)
        print("{}({},{}) called".format(
            fn.__name__, ", ".join(repr(a) for a in args),
            ", ".join(str(k) + "=" + repr(v) for k, v in kwargs.items())))
        print("-->", repr(ret))
        print("-" * 40, "]")
        return ret
    return _tfn


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


def cache_attr(attr):
    """A method decorator that caches the result of a method in an attribute,
    intended for e.g. __str__

    >>> class MyClass():
    ...     def __init__(self):
    ...         self._str = None
    ...
    ...     @cache_attr('_str')
    ...     def __str__(self):
    ...          return "MyClass"
    >>> a = MyClass()
    >>> a._str  # None
    >>> str(a)
    'MyClass'
    >>> a._str
    'MyClass'
    """
    def tie_to_attr_decorator(meth):
        """Decorator that ties `meth` to the fixed `attr`"""
        def tied_method(self):
            if getattr(self, attr) is None:
                setattr(self, attr, meth(self))
            return getattr(self, attr)
        return tied_method
    return tie_to_attr_decorator


class Expression(metaclass=ABCMeta):
    """Abstract class for QNET Expressions. All algebraic objects are either
    scalars (numbers or Sympy expressions) or instances of Expression.

    Expressions should generally be instantiated using the `create` class
    method, which takes into account the algebraic properties of the Expression
    and and applies simplifications. It also uses memoization to cache all
    known (sub-)expression. This is possible because expressions are intended
    to be immutable. Any changes to an expression should be made through e.g.
    :meth:`substitute`, which returns a new modified expression.

    Every expression has a well-defined list of positional and keyword
    arguments that uniquely determine the expression and that may be accessed
    through the `args` and `kwargs` property. That is,

        expr.__class__(*expr.args, **expr.kwargs)

    will return and object identical to `expr`.
    """

    # Note: all subclasses of Exresssion that override __init__ or create
    # *must* call the corresponding superclass method *at the end*. Otherwise,
    # caching will not work correctly

    _rules = []
    _simplifications = []

    # we cache all instances of Expressions for fast construction
    _instances = {}

    # eventually, we should ensure that the create method is idempotent, i.e.
    # expr.create(*expr.args, **expr.kwargs) == expr(*expr.args, **expr.kwargs)
    _create_idempotent = False
    # At this point, match_replace_binary does not yet guarantee this

    def __init__(self, *args, **kwargs):
        # hash, tex, and repr str, generated on demand (lazily)
        self._hash = None
        self._tex = None
        self._repr = None
        self._str = None

    @classmethod
    def create(cls, *args, **kwargs):
        """Instead of directly instantiating, it is recommended to use create,
        which applies simplifications to the args and keyword arguments
        according to the `_simplifications` class attribute, and returns an
        appropriate object (which may or may not be an instance of the original
        class)
        """
        key = cls._instance_key(args, kwargs)
        try:
            return cls._instances[key]
        except KeyError:
            pass
        for simplification in cls._simplifications:
            simplified = simplification(cls, args, kwargs)
            try:
                args, kwargs = simplified
            except (TypeError, ValueError):
                # We assume that if the simplification didn't return a tuple,
                # the result is a fully instantiated object
                cls._instances[key] = simplified
                if cls._create_idempotent:
                    try:
                        key2 = simplified._instance_key(simplified.args,
                                                        simplified.kwargs)
                        if key2 != key:
                            cls._instances[key2] = simplified  # simplified key
                    except AttributeError:
                        #  simplified might e.g. be a scalar and not have
                        #  _instance_key
                        pass
                return simplified
        if len(kwargs) > 0:
            cls._has_kwargs = True
        instance = cls(*args, **kwargs)
        cls._instances[key] = instance
        if cls._create_idempotent:
            key2 = cls._instance_key(args, kwargs)
            if key2 != key:
                cls._instances[key2] = instance  # instantiated key
        return instance

    @classmethod
    def _instance_key(cls, args, kwargs):
        """Function that calculates a unique "key" (a nested tuple) for the
        given args and kwargs. It is the basis of the hash of an Expression,
        and is used for the internal caching of instances.

        Two expressions for which `expr._instance_key(expr.args, expr.kwargs)`
        gives the same result are identical by definition (although `expr1 is
        expr2` is not guaranteed to hold)
        """
        return (cls, tuple(args), tuple(sorted(kwargs.items())))

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
        return (self is other) or hash(self) == hash(other)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._instance_key(self.args, self.kwargs))
        return self._hash

    @cache_attr('_repr')
    def __repr__(self):
        args = self.args
        keys = sorted(self.kwargs.keys())
        kwargs_sep = ''
        if len(self.kwargs) > 0:
            kwargs_sep = ', '
        return (str(self.__class__.__name__) + "(" +
                ", ".join([repr(arg) for arg in args]) + kwargs_sep +
                ", ".join(["%s=%s" % (key, repr(self.kwargs[key]))
                           for key in keys]) + ")")

    @cache_attr('_str')
    def __str__(self):
        return repr(self)

    def substitute(self, var_map):
        """Substitute all_symbols for other expressions.

        Args:
            var_map (dict): Dictionary with entries of the form
                ``{expr: substitution}``
        """
        return self._substitute(var_map)

    def _substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        new_args = [substitute(arg, var_map) for arg in self.args]
        new_kwargs = {key: substitute(val, var_map)
                      for (key, val) in self.kwargs.items()}
        return self.__class__.create(*new_args, **new_kwargs)

    @cache_attr('_tex')
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

    def __getstate__(self):
        """state to be pickled"""
        d = self.__dict__.copy()
        if "_hash" in d:
            del d["_hash"]
        return d


def _str_instance_key(key):
    """Format the key (Expression_instance_key result) as a slightly more
    readable string corresponding to the "create" call.
    """
    args_str = ", ".join([str(arg) for arg in key[1]])
    kw_str = ''
    if len(key[2]) > 0:
        kw_str = ', ' + ", ".join(["%s=%s" % (k, v) for (k, v) in key[2]])
    return key[0].__name__ + ".create(" + args_str + kw_str + ')'


def _print_debug_cache(prefix, color, key, instance, level=0):
    """Routine that can be useful for debugging the Expression instance cache,
    e.g.

        _print_debug_cache('HIT', 'green', key, cls._instances[key])
        _print_debug_cache('store.b', 'red', key, simplified)
    """
    import click
    msg = (click.style("%s: %17x" % (prefix, hash(key)), fg=color) +
           " " + _str_instance_key(key) +
           click.style(" -> %s" % hash(instance), fg=color) +
           " %s" % str(instance))
    click.echo("  " * (level+1) + msg)


def check_idempotent_create(expr):
    """Check that an expression is 'idempotent'"""
    print("*** CHECKING IDEMPOTENCY of %s" % expr)
    if isinstance(expr, Expression):
        new_expr = expr.create(*expr.args, **expr.kwargs)
        if new_expr != expr:
            from IPython.core.debugger import Tracer
            Tracer()()
            print(expr)
            print(new_expr)
    print("*** IDEMPOTENCY OK")


def substitute(expr, var_map):
    """Substitute symbols or (sub-)expressions with the given replacements and
    re-evalute the result

    Args:
        expr: The expression in which to perform the substitution
        var_map (dict): The substitution dictionary.
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
        if isinstance(other, (int, str)):
            return False
        if isinstance(other, KeyTuple):
            return super(KeyTuple, self).__lt__(other)
        raise AlgebraException("Cannot compare: {}".format(other))

    def __gt__(self, other):
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
    """Base class for all "operations", i.e. Expressions that act algebraically
    on other expressions (their "operands").

    Operations differ from more general Expressions by the convention that the
    arguments of the Operator are exactly the operands (which must be members
    of the algebra!) Any other parameters (non-operands) that may be required
    must be given as keyword-arguments.
    """

    def __init__(self, *operands, **kwargs):
        self._operands = operands
        super().__init__(*operands, **kwargs)

    @property
    def operands(self):
        """Tuple of operands of the operation"""
        return self._operands

    @property
    def args(self):
        """Alias for operands"""
        return self._operands



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


def singleton_object(cls):
    """Class decorator that transforms (and replaces) a class definition (which
    must have a Singleton metaclass) with the actual singleton object. Ensures
    that the resulting object can still be "instantiated" (i.e., called),
    returning the same object. Also ensures the object can be pickled, is
    hashable, and has the correct string representation (the name of the
    singleton)
    """
    assert isinstance(cls, Singleton), \
        cls.__name__ + " must use Singleton metaclass"

    def self_instantiate(self):
        return self

    cls.__call__ = self_instantiate
    cls.__hash__ = lambda self: hash(cls)
    cls.__repr__ = lambda self: cls.__name__
    cls.__reduce__ = lambda self: cls.__name__
    return cls()


class Singleton(ABCMeta):
    """Metaclass for singletons. Any instantiation of a Singleton class yields
    the exact same object, e.g.:

    >>> class MyClass(metaclass=Singleton):
    ...     pass
    >>> a = MyClass()
    >>> b = MyClass()
    >>> a is b
    True
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]
