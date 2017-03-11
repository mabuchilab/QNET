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
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################
r"""The abstract algebra package provides the foundation for
symbolic algebra of quantum objects or circuits. All symbolic objects are
either scalars (see :mod:`~qnet.algebra.scalar_types`)  or an instance of
:class:`Expression`. Algebraic combinations of atomic expressions are instances
of :class:`Operation`. In this way, any symbolic expression is a tree of
operations, with children of each node defined through the
:attr:`Operation.operands` attribute, and the leaves being atomic expressions
or scalars.

See :ref:`abstract_algebra` for design details and usage.
"""
from abc import ABCMeta, abstractproperty
from contextlib import contextmanager
from copy import copy
from functools import reduce

from .pattern_matching import (
    ProtoExpr, match_pattern, wc, pattern_head, pattern)
from .singleton import Singleton
from .scalar_types import SCALAR_TYPES
from ..printing import AsciiPrinter, LaTeXPrinter, UnicodePrinter
from ..printing import srepr

__all__ = [
    'AlgebraException', 'AlgebraError', 'CannotSimplify',
    'WrongSignatureError', 'Expression', 'Operation', 'all_symbols',
    'extra_binary_rules', 'extra_rules', 'no_instance_caching', 'no_rules',
    'set_union', 'simplify', 'substitute', 'temporary_instance_cache']

__private__ = [  # anything not in __all__ must be in __private__
    'assoc', 'idem', 'orderby', 'filter_neutral', 'match_replace',
    'match_replace_binary', 'cache_attr', 'check_idempotent_create']


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

    ::

        expr.__class__(*expr.args, **expr.kwargs)

    will return and object identical to `expr`.

    Class attributes:
        instance_caching (bool):  Flag to indicate whether the `create` class
            method should cache the instantiation of instances
    """
    # Note: all subclasses of Exression that override `__init__` or `create`
    # *must* call the corresponding superclass method *at the end*. Otherwise,
    # caching will not work correctly

    # Printer instances handling __str__, __repr__, etc.
    _str_printer = UnicodePrinter  # for __str__()
    _repr_printer = UnicodePrinter  # for __repr__()
    _tex_printer = LaTeXPrinter  # for _tex_()
    _ascii_printer = AsciiPrinter  # for _ascii_()
    _unicode_printer = UnicodePrinter  # for _unicode_()

    # should _ascii_, _unicode_, _tex_ be returned from cache?
    _cached_rendering = True
    # should we force re-rendering of the cached representation?
    _force_cache = False

    _simplifications = []

    # we cache all instances of Expressions for fast construction
    _instances = {}
    instance_caching = True

    # eventually, we should ensure that the create method is idempotent, i.e.
    # expr.create(*expr.args, **expr.kwargs) == expr(*expr.args, **expr.kwargs)
    _create_idempotent = False

    # At this point, match_replace_binary does not yet guarantee this

    def __init__(self, *args, **kwargs):
        # hash, tex, and repr str, generated on demand (lazily) -- see also
        # _cached_rendering class attribute
        self._hash = None
        self._tex = None
        self._ascii = None
        self._unicode = None

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
            if cls.instance_caching:
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
                if cls.instance_caching:
                    cls._instances[key] = simplified
                if cls._create_idempotent and cls.instance_caching:
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
        if cls.instance_caching:
            cls._instances[key] = instance
        if cls._create_idempotent and cls.instance_caching:
            key2 = cls._instance_key(args, kwargs)
            if key2 != key:
                cls._instances[key2] = instance  # instantiated key
        return instance

    @classmethod
    def _instance_key(cls, args, kwargs):
        """Function that calculates a unique "key" (as a tuple) for the
        given args and kwargs. It is the basis of the hash of an Expression,
        and is used for the internal caching of instances.

        Two expressions for which `expr._instance_key(expr.args, expr.kwargs)`
        gives the same result are identical by definition (although `expr1 is
        expr2` is not guaranteed to hold)
        """
        return (cls,) + tuple(args) + tuple(sorted(kwargs.items()))

    @abstractproperty
    def args(self):
        """The tuple of positional arguments for the instantiation of the
        Expression"""
        raise NotImplementedError(self.__class__.__name__)

    @property
    def kwargs(self):
        """The dictionary of keyword-only arguments for the instantiation of
        the Expression"""
        # Subclasses must override this property if and only if they define
        # keyword-only arguments in their __init__ method
        if hasattr(self, '_has_kwargs') and self._has_kwargs:
            raise NotImplementedError(
                "Class %s does not provide a kwargs property"
                % str(self.__class__.__name__))
        return {}

    @property
    def minimal_kwargs(self):
        """A "minimal" dictionary of keyword-only arguments, i.e. a subsect of
        `kwargs` that may exclude default options"""
        return self.kwargs

    def __eq__(self, other):
        return (self is other) or hash(self) == hash(other)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._instance_key(self.args, self.kwargs))
        return self._hash

    def __repr__(self):
        return self._repr_printer.render(self)

    def __str__(self):
        return self._str_printer.render(self)

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
        if isinstance(self.__class__, Singleton):
            return self
        new_args = [substitute(arg, var_map) for arg in self.args]
        new_kwargs = {key: substitute(val, var_map)
                      for (key, val) in self.kwargs.items()}
        return self.__class__.create(*new_args, **new_kwargs)

    def simplify(self, rules=None):
        """Recursively re-instantiate the expression, while applying all of the
        given `rules` to all encountered (sub-) expressions"""
        if rules is None:
            rules = []
        new_args = [simplify(arg, rules) for arg in self.args]
        new_kwargs = {key: simplify(val, rules)
                      for (key, val) in self.kwargs.items()}
        simplified = self.__class__.create(*new_args, **new_kwargs)
        for (rule, replacement) in rules:
            matched = rule.match(simplified)
            if matched:
                try:
                    return replacement(**matched)
                except CannotSimplify:
                    pass
        return simplified

    def _render(self, fmt, adjoint=False):
        printer = getattr(self, "_" + fmt + "_printer")
        if adjoint:
            raise NotImplementedError("_render is defined for adjoint=True")
            # Any _render that falls back to head_repr should never be called
            # in a context that would require the adjoint
        return printer.render_head_repr(self)

    def _cached_render(self, fmt, adjoint=False):
        if adjoint:
            return self._render(fmt, adjoint=True)
        if self._cached_rendering:
            attr = '_' + fmt
            if self._force_cache:
                setattr(self, attr, None)
            if getattr(self, attr) is None:
                setattr(self, attr, self._render(fmt))
            return getattr(self, attr)
        else:
            return self._render(fmt)

    def _tex_(self, adjoint=False):
        return self._cached_render('tex', adjoint=adjoint)

    def _ascii_(self, adjoint=False):
        return self._cached_render('ascii', adjoint=adjoint)

    def _unicode_(self, adjoint=False):
        return self._cached_render('unicode', adjoint=adjoint)

    def _repr_latex_(self):
        """For compatibility with the IPython notebook, generate TeX expression
        and surround it with $'s.
        """
        return "$" + self._tex_() + "$"

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
    click.echo("  " * (level + 1) + msg)


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


def simplify(expr, rules=None):
    """Recursively re-instantiate the expression, while applying all of the
    given `rules` to all encountered (sub-) expressions

    Args:
        expr:  Any Expression or scalar object
        rules (list): A list of tuples ``(pattern, replacement)`` where `rule`
            is an instance of :class:`Pattern`) and `replacement` is a
            callable.  The pattern will be matched against any expression that
            is encountered during the re-instantiation. If the `pattern`
            matches, then the (sub-)expression is replaced by the result of
            calling `replacement` while passing any wildcards from `pattern` as
            keyword arguments. If `replacement` raises `CannotSimplify`, it
            will be ignored

    Note:
        Instead of or in addition to passing `rules`, `simplify` can often be
        combined with e.g. `extra_rules` / `extra_binary_rules` context
        managers. If a simplification can be handled through these context
        managers, this is usually more efficient than an equivalent rule.
        However, both really are complemetary: the rules defined in the context
        managers are applied *before* instantation (hence these these patterns
        are instantiated through `pattern_head`). In contrast, the patterns
        defined in `rules` are applied against instantiated expressions.
    """
    if rules is None:
        rules = []
    if isinstance(expr, Expression):
        return expr.simplify(rules)
    else:
        for (rule, replacement) in rules:
            matched = rule.match(expr)
            if matched:
                try:
                    return replacement(**matched)
                except CannotSimplify:
                    pass
        return expr


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
        ...     order_key = lambda val: val
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
        ...     order_key = lambda val: val
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

        >>> print(srepr(Invert.create("hallo")))  # matches no rule
        Invert('hallo')
        >>> Invert.create(Invert("hallo"))        # matches first rule
        'hallo'
        >>> Invert.create(.2)                     # matches second rule
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
        ...     _simplifications = [match_replace_binary, assoc]
        ...     neutral_element = 0
        >>> FilterDupes.create(1,2,3,4)         # No duplicates
        FilterDupes(1, 2, 3, 4)
        >>> FilterDupes.create(1,2,2,3,4)       # Some duplicates
        FilterDupes(1, 2, 3, 4)

    Note that this only works for *subsequent* duplicate entries:

        >>> FilterDupes.create(1,2,3,2,4)       # No *subsequent* duplicates
        FilterDupes(1, 2, 3, 2, 4)

    Any operation that uses binary reduction must be associative and define a
    neutral element. The binary rules must be compatible with associativity,
    i.e. there is no specific order in which the rules are applied to pairs of
    operands.
    """
    assert assoc in cls._simplifications, (
        cls.__name__ + " must be associative to use match_replace_binary")
    assert hasattr(cls, 'neutral_element'), (
        cls.__name__ + " must define a neutral element to use "
                       "match_replace_binary")
    fops = _match_replace_binary(cls, list(ops))
    if len(fops) == 1:
        return fops[0]
    elif len(fops) == 0:
        return cls.neutral_element
    else:
        return fops, kwargs


def _match_replace_binary(cls, ops: list) -> list:
    """Reduce list of `ops`"""
    n = len(ops)
    if n <= 1:
        return ops
    ops_left = ops[:n // 2]
    ops_right = ops[n // 2:]
    return _match_replace_binary_combine(
        cls,
        _match_replace_binary(cls, ops_left),
        _match_replace_binary(cls, ops_right))


def _match_replace_binary_combine(cls, a: list, b: list) -> list:
    """combine two fully reduced lists a, b"""
    if len(a) == 0 or len(b) == 0:
        return a + b
    r = _get_binary_replacement(a[-1], b[0], cls._binary_rules)
    if r is None:
        return a + b
    if r == cls.neutral_element:
        return _match_replace_binary_combine(cls, a[:-1], b[1:])
    if isinstance(r, cls):
        r = list(r.args)
    else:
        r = [r, ]
    return _match_replace_binary_combine(
        cls,
        _match_replace_binary_combine(cls, a[:-1], r),
        b[1:])


###############################################################################
############################## CONTEXT MANAGERS ###############################
###############################################################################


@contextmanager
def no_instance_caching():
    """Temporarily disable the caching of instances through
    :meth:`Expression.create`
    """
    # this assumes that no sub-class of Expression shadows
    # Expression.instance_caching
    orig_flag = Expression.instance_caching
    Expression.instance_caching = False
    yield
    Expression.instance_caching = orig_flag


@contextmanager
def temporary_instance_cache(cls):
    """Use a temporary cache for instances obtained from the `create` method of
    the given `cls`. That is, no cached instances from outside of the managed
    context will be used within the managed context, and vice versa"""
    orig_instances = cls._instances
    cls._instances = {}
    yield
    cls._instances = orig_instances


@contextmanager
def extra_rules(cls, rules):
    """Context manager that temporarily adds the given rules to `cls` (to be
    processed by `match_replace`. Implies `temporary_instance_cache`.
    """
    orig_rules = copy(cls._rules)
    cls._rules.extend(rules)
    orig_instances = cls._instances
    cls._instances = {}
    yield
    cls._rules = orig_rules
    cls._instances = orig_instances


@contextmanager
def extra_binary_rules(cls, rules):
    """Context manager that temporarily adds the given rules to `cls` (to be
    processed by `match_replace_binary`. Implies `temporary_instance_cache`.
    """
    orig_rules = copy(cls._binary_rules)
    cls._binary_rules.extend(rules)
    orig_instances = cls._instances
    cls._instances = {}
    yield
    cls._binary_rules = orig_rules
    cls._instances = orig_instances


@contextmanager
def no_rules(cls):
    """Context manager that temporarily disables all rules (processed by
    `match_replace` or `match_replace_binary`) for the given `cls`. Implies
    `temporary_instance_cache`.
    """
    has_rules = True
    has_binary_rules = True
    orig_instances = cls._instances
    cls._instances = {}
    try:
        orig_rules = cls._rules
        cls._rules = []
    except AttributeError:
        has_rules = False
    try:
        orig_binary_rules = cls._binary_rules
        cls._binary_rules = []
    except AttributeError:
        has_binary_rules = False
    yield
    if has_rules:
        cls._rules = orig_rules
    if has_binary_rules:
        cls._binary_rules = orig_binary_rules
    cls._instances = orig_instances
