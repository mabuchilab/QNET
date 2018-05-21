r"""Base classes for all Expressions and Operations.

The abstract algebra package provides the foundation for
symbolic algebra of quantum objects or circuits. All symbolic objects are
an instance of :class:`Expression`. Algebraic combinations of atomic
expressions are instances of :class:`Operation`. In this way, any symbolic
expression is a tree of operations, with children of each node defined through
the :attr:`Operation.operands` attribute, and the leaves being atomic
expressions.

See :ref:`abstract_algebra` for design details and usage.
"""
import logging
from abc import ABCMeta, abstractmethod

from sympy import (
    Basic as SympyBasic)
from sympy.core.sympify import SympifyError

from .exceptions import CannotSimplify
from ..pattern_matching import ProtoExpr, pattern
from ...utils.singleton import Singleton

__all__ = [
    'Expression', 'Operation', 'simplify', 'simplify_by_method', 'substitute']

__private__ = []  # anything not in __all__ must be in __private__

LEVEL = 0  # for debugging create method

LOG = False  # emit debug logging messages?
LOG_NO_MATCH = False  # also log non-matching rules? (very verbose!)
# Note: you may manually set the above variables to True for debugging. Some
# tests (e.g. the tests for the algebraic rules) will also automatically
# activate this logging functionality, as they rely on inspecting the debug
# messages from object creation.


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

    _simplifications = []

    # we cache all instances of Expressions for fast construction
    _instances = {}
    instance_caching = True

    # eventually, we should ensure that the create method is idempotent, i.e.
    # expr.create(*expr.args, **expr.kwargs) == expr(*expr.args, **expr.kwargs)
    _create_idempotent = False

    # At this point, match_replace_binary does not yet guarantee this

    def __init__(self, *args, **kwargs):
        self._hash = None
        self._instance_key = self._get_instance_key(args, kwargs)

    @classmethod
    def create(cls, *args, **kwargs):
        """Instead of directly instantiating, it is recommended to use create,
        which applies simplifications to the args and keyword arguments
        according to the `_simplifications` class attribute, and returns an
        appropriate object (which may or may not be an instance of the original
        class)
        """
        global LEVEL
        if LOG:
            logger = logging.getLogger(__name__ + '.create')
            logger.debug(
                "%s%s.create(*args, **kwargs); args = %s, kwargs = %s",
                ("  " * LEVEL), cls.__name__, args, kwargs)
            LEVEL += 1
        key = cls._get_instance_key(args, kwargs)
        try:
            if cls.instance_caching:
                instance = cls._instances[key]
                if LOG:
                    LEVEL -= 1
                    logger.debug("%s(cached)-> %s", ("  " * LEVEL), instance)
                return instance
        except KeyError:
            pass
        for i, simplification in enumerate(cls._simplifications):
            if LOG:
                try:
                    simpl_name = simplification.__name__
                except AttributeError:
                    simpl_name = "simpl%d" % i
            simplified = simplification(cls, args, kwargs)
            try:
                args, kwargs = simplified
                if LOG:
                    logger.debug(
                        "%s(%s)-> args = %s, kwargs = %s", ("  " * LEVEL),
                        simpl_name, args, kwargs)
            except (TypeError, ValueError):
                # We assume that if the simplification didn't return a tuple,
                # the result is a fully instantiated object
                if cls.instance_caching:
                    cls._instances[key] = simplified
                if cls._create_idempotent and cls.instance_caching:
                    try:
                        key2 = simplified._instance_key
                        if key2 != key:
                            cls._instances[key2] = simplified  # simplified key
                    except AttributeError:
                        #  simplified might e.g. be a scalar and not have
                        #  _instance_key
                        pass
                if LOG:
                    LEVEL -= 1
                    logger.debug(
                        "%s(%s)-> %s", ("  " * LEVEL), simpl_name, simplified)
                return simplified
        if len(kwargs) > 0:
            cls._has_kwargs = True
        instance = cls(*args, **kwargs)
        if cls.instance_caching:
            cls._instances[key] = instance
        if cls._create_idempotent and cls.instance_caching:
            key2 = cls._get_instance_key(args, kwargs)
            if key2 != key:
                cls._instances[key2] = instance  # instantiated key
        if LOG:
            LEVEL -= 1
            logger.debug("%s -> %s", ("  " * LEVEL), instance)
        return instance

    @classmethod
    def _get_instance_key(cls, args, kwargs):
        """Function that calculates a unique "key" (as a tuple) for the
        given args and kwargs. It is the basis of the hash of an Expression,
        and is used for the internal caching of instances. Every Expression
        stores this key in the `_instance_key` attribute.

        Two expressions for which `expr._instance_key` is the same are
        identical by definition (although `expr1 is expr2` generally only holds
        for explicit Singleton instances)
        """
        return (cls,) + tuple(args) + tuple(sorted(kwargs.items()))

    @property
    @abstractmethod
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
        """A "minimal" dictionary of keyword-only arguments, i.e. a subset of
        `kwargs` that may exclude default options"""
        return self.kwargs

    def __eq__(self, other):
        try:
            return ((self is other) or
                    (self._instance_key == other._instance_key))
        except AttributeError:
            return False

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._instance_key)
        return self._hash

    def __repr__(self):
        # This method will be replaced by init_printing()
        from qnet.printing import init_printing
        init_printing()
        return repr(self)

    def __str__(self):
        # This method will be replaced by init_printing()
        from qnet.printing import init_printing
        init_printing()
        return str(self)

    def substitute(self, var_map):
        """Substitute sub-expressions

        Args:
            var_map (dict): Dictionary with entries of the form
                ``{expr: substitution}``
        """
        return self._substitute(var_map)

    def _substitute(self, var_map, safe=False):
        """Implementation of :meth:`substitute`.

        For internal use, the `safe` keyword argument allows to perform a
        substitution on the `args` and `kwargs` of the expression only,
        guaranteeing that the type of the expression does not change, at the
        cost of possibly not returning a maximally simplified expression. The
        `safe` keyword is not handled recursively, i.e. any `args`/`kwargs`
        will be fully simplified, possibly changing their types.
        """
        if self in var_map:
            if not safe or (type(var_map[self]) == type(self)):
                return var_map[self]
        if isinstance(self.__class__, Singleton):
            return self
        new_args = [substitute(arg, var_map) for arg in self.args]
        new_kwargs = {key: substitute(val, var_map)
                      for (key, val) in self.kwargs.items()}
        if safe:
            return self.__class__(*new_args, **new_kwargs)
        else:
            return self.create(*new_args, **new_kwargs)

    def simplify(self, rules=None):
        """Recursively re-instantiate the expression, while applying all of the
        given `rules` to all encountered (sub-) expressions

        Args:
            rules (list or ~collections.OrderedDict): List of rules or
                dictionary mapping names to rules, where each rule is a tuple
                (Pattern, replacement callable)
        """
        if rules is None:
            rules = {}
        new_args = [simplify(arg, rules) for arg in self.args]
        new_kwargs = {key: simplify(val, rules)
                      for (key, val) in self.kwargs.items()}
        simplified = self.create(*new_args, **new_kwargs)
        try:
            # `rules` is an OrderedDict key => (pattern, replacement)
            items = rules.items()
        except AttributeError:
            # `rules` is a list of (pattern, replacement) tuples
            items = enumerate(rules)
        for key, (pat, replacement) in items:
            matched = pat.match(simplified)
            if matched:
                try:
                    return replacement(**matched)
                except CannotSimplify:
                    pass
        return simplified

    def _repr_latex_(self):
        """For compatibility with the IPython notebook, generate TeX expression
        and surround it with $'s.
        """
        # This method will be replaced by init_printing()
        from qnet.printing import init_printing
        init_printing()
        return self._repr_latex_()

    def _sympy_(self):
        # By default, when a QNET expression occurring in a SymPy context (e.g.
        # when converting a QNET Matrix to a Sympy Matrix), sympify will try to
        # parse the string representation of the Expression. This will usually
        # fail, but when it doesn't, it always produces nonsense. Thus, we make
        # it fail explicitly
        raise SympifyError("QNET expressions cannot be converted to SymPy")

    @property
    def free_symbols(self):
        """Set of free SymPy symbols contained within the expression."""
        res = set()
        return res.union(*[_free_symbols(arg) for arg in self.args])

    @property
    def bound_symbols(self):
        """Set of bound SymPy symbols in the expression"""
        res = set()
        return res.union(*[_bound_symbols(arg) for arg in self.args])

    def __ne__(self, other):
        """If it is well-defined (i.e. boolean), simply return
        the negation of ``self.__eq__(other)``
        Otherwise return NotImplemented.
        """
        eq = self.__eq__(other)
        if type(eq) is bool:
            return not eq
        return NotImplemented


def substitute(expr, var_map):
    """Substitute symbols or (sub-)expressions with the given replacements and
    re-evalute the result

    Args:
        expr: The expression in which to perform the substitution
        var_map (dict): The substitution dictionary.
    """
    try:
        if isinstance(expr, SympyBasic):
            sympy_var_map = {
                k: v for (k, v) in var_map.items()
                if isinstance(k, SympyBasic)}
            return expr.subs(sympy_var_map)
        else:
            return expr.substitute(var_map)
    except AttributeError:
        if expr in var_map:
            return var_map[expr]
        return expr


def _simplify_expr(expr, rules=None):
    """Non-recursively match expr again all rules"""
    if rules is None:
        rules = {}
    try:
        # `rules` is an OrderedDict key => (pattern, replacement)
        items = rules.items()
    except AttributeError:
        # `rules` is a list of (pattern, replacement) tuples
        items = enumerate(rules)
    for key, (pat, replacement) in items:
        matched = pat.match(expr)
        if matched:
            try:
                return replacement(**matched)
            except CannotSimplify:
                pass
    return expr


def simplify(expr, rules=None):
    """Recursively re-instantiate the expression, while applying all of the
    given `rules` to all encountered (sub-) expressions

    Args:
        expr:  Any Expression or scalar object
        rules (list, ~collections.OrderedDict): A list of rules dictionary
            mapping names to rules, where each rule is a tuple ``(pattern,
            replacement)`` where `pattern` is an instance of :class:`.Pattern`)
            and `replacement` is a callable. The pattern will be matched
            against any expression that is encountered during the
            re-instantiation. If the `pattern` matches, then the
            (sub-)expression is replaced by the result of calling `replacement`
            while passing any wildcards from `pattern` as keyword arguments. If
            `replacement` raises :exc:`.CannotSimplify`, it will be ignored

    Note:
        Instead of or in addition to passing `rules`, `simplify` can often be
        combined with e.g. `extra_rules` / `extra_binary_rules` context
        managers. If a simplification can be handled through these context
        managers, this is usually more efficient than an equivalent rule.
        However, both really are complementary: the rules defined in the
        context managers are applied *before* instantiation (hence these these
        patterns are instantiated through `pattern_head`). In contrast, the
        patterns defined in `rules` are applied against instantiated
        expressions.
    """
    if LOG:
        logger = logging.getLogger(__name__ + '.simplify')
    if rules is None:
        rules = {}
    stack = []
    path = []
    if isinstance(expr, Expression):
        stack.append(ProtoExpr.from_expr(expr))
        path.append(0)
        if LOG:
            logger.debug(
                "Starting at level 1: placing expr on stack: %s", expr)
        while True:
            i = path[-1]
            try:
                arg = stack[-1][i]
                if LOG:
                    logger.debug(
                        "At level %d: considering arg %d: %s",
                        len(stack), i+1, arg)
            except IndexError:
                # done at this level
                path.pop()
                expr = stack.pop().instantiate()
                expr = _simplify_expr(expr, rules)
                if len(stack) == 0:
                    if LOG:
                        logger.debug(
                            "Complete level 1: returning simplified expr: %s",
                            expr)
                    return expr
                else:
                    stack[-1][path[-1]] = expr
                    path[-1] += 1
                    if LOG:
                        logger.debug(
                            "Complete level %d. At level %d, setting arg %d "
                            "to simplified expr: %s", len(stack)+1, len(stack),
                            path[-1], expr)
            else:
                if isinstance(arg, Expression):
                    stack.append(ProtoExpr.from_expr(arg))
                    path.append(0)
                    if LOG:
                        logger.debug("   placing arg on stack")
                else:  # scalar
                    stack[-1][i] = _simplify_expr(arg, rules)
                    if LOG:
                        logger.debug(
                            "   arg is leaf, replacing with simplified expr: "
                            "%s", stack[-1][i])
                    path[-1] += 1
    else:
        return _simplify_expr(expr, rules)


def simplify_by_method(expr, *method_names, head=None, **kwargs):
    """Simplify `expr` by calling all of the given methods on it, if possible.

    Args:
        expr: The expression to simplify
        method_names: One or more method names. Any subexpression that has a
            method with any of the `method_names` will be replaced by the
            result of calling the method.
        head (None or type or list): An optional list of classes to which the
            simplification should be restricted
        kwargs: keyword arguments to be passed to all methods

    Note:
        If giving multiple `method_names`, all the methods must take all of the
        `kwargs`
    """

    method_names_set = set(method_names)

    def has_any_wanted_method(expr):
        return len(method_names_set.intersection(dir(expr))) > 0

    def apply_methods(expr, method_names, **kwargs):
        for mtd in method_names:
            if hasattr(expr, mtd):
                try:
                    expr = getattr(expr, mtd)(**kwargs)
                except TypeError:
                    # mtd turns out to not actually be a method (not callable)
                    pass
        return expr

    pat = pattern(head=head, wc_name='X', conditions=(has_any_wanted_method, ))

    return simplify(
        expr, [(pat, lambda X: apply_methods(X, method_names, **kwargs))])


def _free_symbols(expr):
    try:
        return expr.free_symbols
    except AttributeError:
        return set()


def _bound_symbols(expr):
    try:
        return expr.bound_symbols
    except AttributeError:
        return set()


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
