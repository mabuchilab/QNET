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
import inspect
import textwrap
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from sympy import (
    Basic as SympyBasic)
from sympy.core.sympify import SympifyError

from .exceptions import CannotSimplify
from ..pattern_matching import ProtoExpr
from ...utils.singleton import Singleton
from ...utils.containers import nested_tuple

__all__ = [
    'Expression', 'Operation', 'substitute']

__private__ = []  # anything not in __all__ must be in __private__

LEVEL = 0  # for debugging create method

LOG = False  # emit debug logging messages?
LOG_NO_MATCH = False  # also log non-matching rules? (very verbose!)
# Note: you may manually set the above variables to True for debugging. Some
# tests (e.g. the tests for the algebraic rules) will also automatically
# activate this logging functionality, as they rely on inspecting the debug
# messages from object creation.


class Expression(metaclass=ABCMeta):
    """Base class for all QNET Expressions

    Expressions should generally be instantiated using the :meth:`create` class
    method, which takes into account the algebraic properties of the Expression
    and and applies simplifications. It also uses memoization to cache all
    known (sub-)expression. This is possible because expressions are intended
    to be immutable. Any changes to an expression should be made through e.g.
    :meth:`substitute` or :meth:`apply_rule`, which returns a new modified
    expression.

    Every expression has a well-defined list of positional and keyword
    arguments that uniquely determine the expression and that may be accessed
    through the :attr:`args` and :attr:`kwargs` property. That is,

    ::

        expr.__class__(*expr.args, **expr.kwargs)

    will return and object identical to `expr`.

    Class attributes:
        instance_caching (bool):  Flag to indicate whether the :meth:`create`
            class method should cache the instantiation of instances. If True,
            repeated calls to :meth:`create` with the same arguments return
            instantly, instead of re-evaluating all simplifications and rules.
        simplifications (list): List of callable simplifications that
            :meth:`create` will use to process its positional and keyword
            arguments. Each callable must take three parameters (the class, the
            list `args` of positional arguments given to :meth:`create` and a
            dictionary `kwargs` of keyword arguments given to :meth:`create`)
            and return either a tuple of new `args` and `kwargs` (which are
            then handed to the next callable), or an :class:`Expression` (which
            is directly returned as the result of the call to :meth:`create`).
            The built-in available simplification callables are in
            :mod:`~qnet.algebra.core.algebraic_properties`
    """
    # Note: all subclasses of Exression that override `__init__` or `create`
    # *must* call the corresponding superclass method *at the end*. Otherwise,
    # caching will not work correctly

    simplifications = []

    # we cache all instances of Expressions for fast construction
    _instances = {}
    instance_caching = True

    # eventually, we should ensure that the create method is idempotent, i.e.
    # expr.create(*expr.args, **expr.kwargs) == expr(*expr.args, **expr.kwargs)
    _create_idempotent = False

    # At this point, match_replace_binary does not yet guarantee this

    def __init__(self, *args, **kwargs):
        self._hash = None
        self._free_symbols = None
        self._bound_symbols = None
        self._all_symbols = None
        self._instance_key = self._get_instance_key(args, kwargs)

    @classmethod
    def create(cls, *args, **kwargs):
        """Instantiate while applying automatic simplifications

        Instead of directly instantiating `cls`, it is recommended to use
        :meth:`create`, which applies simplifications to the args and keyword
        arguments according to the :attr:`simplifications` class attribute, and
        returns an appropriate object (which may or may not be an instance of
        the original `cls`).

        Two simplifications of particular importance are :func:`.match_replace`
        and :func:`.match_replace_binary` which apply rule-based
        simplifications.

        The :func:`.temporary_rules` context manager may be used to allow
        temporary modification of the automatic simplifications that
        :meth:`create` uses, in particular the rules for
        :func:`.match_replace` and :func:`.match_replace_binary`. Inside the
        managed context, the :attr:`simplifications` class attribute may be
        modified and rules can be managed with :meth:`add_rule` and
        :meth:`del_rules`.
        """
        global LEVEL
        if LOG:
            logger = logging.getLogger('QNET.create')
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
        for i, simplification in enumerate(cls.simplifications):
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
        return (cls,) + nested_tuple(args) + nested_tuple(kwargs)

    @classmethod
    def _rules_attr(cls):
        """Return the name of the attribute with rules for :meth:`create`"""
        from qnet.algebra.core.algebraic_properties import (
            match_replace, match_replace_binary)
        if match_replace in cls.simplifications:
            return '_rules'
        elif match_replace_binary in cls.simplifications:
            return '_binary_rules'
        else:
            raise TypeError(
                "class %s does not have match_replace or "
                "match_replace_binary in its simplifications" % cls.__name__)

    @classmethod
    def add_rule(cls, name, pattern, replacement, attr=None):
        """Add an algebraic rule for :meth:`create` to the class

        Args:
            name (str): Name of the rule. This is used for debug logging to
                allow an analysis of which rules where applied when creating an
                expression. The `name` can be arbitrary, but it must be unique.
                Built-in rules have names ``'Rxxx'`` where ``x`` is a digit
            pattern (.Pattern): A pattern constructed by :func:`.pattern_head`
                to match a :class:`.ProtoExpr`
            replacement (callable): callable that takes the wildcard names
                defined in `pattern` as keyword arguments and returns an
                evaluated expression.
            attr (None or str): Name of the class attribute to which to add the
                rule. If None, one of ``'_rules'``, ``'_binary_rules'`` is
                automatically chosen

        Raises:
            TypeError: if `name` is not a :class:`str` or `pattern` is not a
                :class:`.Pattern` instance
            ValueError: if `pattern` is not set up to match a
                :class:`.ProtoExpr`; if there there is already a rule with the
                same `name`; if `replacement` is not a callable or does not
                take all the wildcard names in `pattern` as arguments
            AttributeError: If invalid `attr`

        Note:
            The "automatic" rules added by this method are applied *before*
            expressions are instantiated (against a corresponding
            :class:`.ProtoExpr`). In contrast,
            :meth:`apply_rules`/:meth:`apply_rule` are applied to fully
            instantiated objects.

            The :func:`.temporary_rules` context manager may be used to create
            a context in which rules may be defined locally.
        """
        from qnet.utils.check_rules import check_rules_dict
        if attr is None:
            attr = cls._rules_attr()
        if name in getattr(cls, attr):
            raise ValueError(
                "Duplicate key '%s': rule already exists" % name)
        getattr(cls, attr).update(check_rules_dict(
            [(name, (pattern, replacement))]))

    @classmethod
    def show_rules(cls, *names, attr=None):
        """Print algebraic rules used by :class:`create`

        Print a summary of the algebraic rules with the given names, or all
        rules if not names a given.

        Args:
            names (str): Names of rules to show
            attr (None or str): Name of the class attribute from which to get
                the rules. Cf. :meth:`add_rule`.

        Raises:
            AttributeError: If invalid `attr`
        """
        from qnet.printing import srepr
        try:
            if attr is None:
                attr = cls._rules_attr()
            rules = getattr(cls, attr)
        except TypeError:
            rules = {}
        for (name, rule) in rules.items():
            if len(names) > 0 and name not in names:
                continue
            pat, repl = rule
            print(name)
            print("    PATTERN:")
            print(textwrap.indent(
                textwrap.dedent(srepr(pat, indented=True)),
                prefix=" "*8))
            print("    REPLACEMENT:")
            print(textwrap.indent(
                textwrap.dedent(inspect.getsource(repl).rstrip()),
                prefix=" "*8))

    @classmethod
    def del_rules(cls, *names, attr=None):
        """Delete algebraic rules used by :meth:`create`

        Remove the rules with the given `names`, or all rules if no names are
        given

        Args:
            names (str): Names of rules to delete
            attr (None or str): Name of the class attribute from which to
                delete the rules. Cf. :meth:`add_rule`.

        Raises:
            KeyError: If any rules in `names` does not exist
            AttributeError: If invalid `attr`
        """
        if attr is None:
            attr = cls._rules_attr()
        if len(names) == 0:
            getattr(cls, attr)  # raise AttributeError if wrong attr
            setattr(cls, attr, OrderedDict())
        else:
            for name in names:
                del getattr(cls, attr)[name]

    @classmethod
    def rules(cls, attr=None):
        """Iterable of rule names used by :meth:`create`

        Args:
            attr (None or str): Name of the class attribute to which to get the
                names. If None, one of ``'_rules'``, ``'_binary_rules'`` is
                automatically chosen
        """
        try:
            if attr is None:
                attr = cls._rules_attr()
            return getattr(cls, attr).keys()
        except TypeError:
            return ()

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
        if self in var_map:
            return var_map[self]
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

    def doit(self, classes=None, recursive=True, **kwargs):
        """Rewrite (sub-)expressions in a more explicit form

        Return a modified expression that is more explicit than the original
        expression. The definition of "more explicit" is decided by the
        relevant subclass, e.g. a :meth:`Commutator <.Commutator.doit>` is
        written out according to its definition.

        Args:
            classes (None or list): an optional list of classes. If given,
                only (sub-)expressions that an instance of one of the classes
                in the list will be rewritten.
            recursive (bool): If True, also rewrite any sub-expressions of any
                rewritten expression. Note that :meth:`doit` always recurses
                into sub-expressions of expressions not affected by it.
            kwargs: Any remaining keyword arguments may be used by the
                :meth:`doit` method of a particular expression.

        Example:

            Consider the following expression::

                >>> from sympy import IndexedBase
                >>> i = IdxSym('i'); N = symbols('N')
                >>> Asym, Csym = symbols('A, C', cls=IndexedBase)
                >>> A = lambda i: OperatorSymbol(StrLabel(Asym[i]), hs=0)
                >>> B = OperatorSymbol('B', hs=0)
                >>> C = lambda i: OperatorSymbol(StrLabel(Csym[i]), hs=0)
                >>> def show(expr):
                ...     print(unicode(expr, show_hs_label=False))
                >>> expr = Sum(i, 1, 3)(Commutator(A(i), B) + C(i)) / N
                >>> show(expr)
                1/N (∑_{i=1}^{3} (Ĉ_i + [Â_i, B̂]))

            Calling :meth:`doit` without parameters rewrites both the indexed
            sum and the commutator::

                >>> show(expr.doit())
                1/N (Ĉ₁ + Ĉ₂ + Ĉ₃ + Â₁ B̂ + Â₂ B̂ + Â₃ B̂ - B̂ Â₁ - B̂ Â₂ - B̂ Â₃)

            A non-recursive call only expands the sum, as it does not recurse
            into the expanded summands::

                >>> show(expr.doit(recursive=False))
                1/N (Ĉ₁ + Ĉ₂ + Ĉ₃ + [Â₁, B̂] + [Â₂, B̂] + [Â₃, B̂])

            We can selectively expand only the sum or only the commutator::

                >>> show(expr.doit(classes=[IndexedSum]))
                1/N (Ĉ₁ + Ĉ₂ + Ĉ₃ + [Â₁, B̂] + [Â₂, B̂] + [Â₃, B̂])

                >>> show(expr.doit(classes=[Commutator]))
                1/N (∑_{i=1}^{3} (Ĉ_i - B̂ Â_i + Â_i B̂))

            Also we can pass a keyword argument that expands the sum only to
            the 2nd term, as documented in :meth:`.Commutator.doit`

                >>> show(expr.doit(classes=[IndexedSum], max_terms=2))
                1/N (Ĉ₁ + Ĉ₂ + [Â₁, B̂] + [Â₂, B̂])
        """
        in_classes = (
            (classes is None) or
            any([isinstance(self, cls) for cls in classes]))
        if in_classes:
            new = self._doit(**kwargs)
        else:
            new = self
        if (new == self) or recursive:
            new_args = []
            for arg in new.args:
                if isinstance(arg, Expression):
                    new_args.append(arg.doit(
                        classes=classes, recursive=recursive, **kwargs))
                else:
                    new_args.append(arg)
            new_kwargs = OrderedDict([])
            for (key, val) in new.kwargs.items():
                if isinstance(val, Expression):
                    new_kwargs[key] = val.doit(
                        classes=classes, recursive=recursive, **kwargs)
                else:
                    new_kwargs[key] = val
            new = new.__class__.create(*new_args, **new_kwargs)
            if new != self and recursive:
                new = new.doit(classes=classes, recursive=True, **kwargs)
        return new

    def _doit(self, **kwargs):
        """Non-recursively rewrite expression in a more explicit form"""
        # Any subclass that overrides :meth:`_doit` should also override
        # :meth:`doit` with a stub (calling ``super().doit`` only), but
        # also provide the documentation for :meth:`_doit` (since :meth:`_doit`
        # won't be rendered by Sphinx)
        return self

    def apply(self, func, *args, **kwargs):
        """Apply `func` to expression.

        Equivalent to ``func(self, *args, **kwargs)``. This method exists for
        easy chaining::

            >>> A, B, C, D = (
            ...     OperatorSymbol(s, hs=1) for s in ('A', 'B', 'C', 'D'))
            >>> expr = (
            ...     Commutator(A * B, C * D)
            ...     .apply(lambda expr: expr**2)
            ...     .apply(expand_commutators_leibniz, expand_expr=False)
            ...     .substitute({A: IdentityOperator}))
        """
        return func(self, *args, **kwargs)

    def apply_rules(self, rules, recursive=True):
        """Rebuild the expression while applying a list of rules

        The rules are applied against the instantiated expression, and any
        sub-expressions if `recursive` is True. Rule application is best though
        of as a pattern-based substitution. This is different from the
        *automatic* rules that :meth:`create` uses (see :meth:`add_rule`),
        which are applied *before* expressions are instantiated.

        Args:
            rules (list or ~collections.OrderedDict): List of rules or
                dictionary mapping names to rules, where each rule is a tuple
                (:class:`Pattern`, replacement callable), cf.
                :meth:`apply_rule`
            recursive (bool): If true (default), apply rules to all arguments
                and keyword arguments of the expression. Otherwise, only the
                expression itself will be re-instantiated.

        If `rules` is a dictionary, the keys (rules names) are used only for
        debug logging, to allow an analysis of which rules lead to the final
        form of an expression.
        """
        if recursive:
            new_args = [_apply_rules(arg, rules) for arg in self.args]
            new_kwargs = {
                key: _apply_rules(val, rules)
                for (key, val) in self.kwargs.items()}
        else:
            new_args = self.args
            new_kwargs = self.kwargs
        simplified = self.create(*new_args, **new_kwargs)
        return _apply_rules_no_recurse(simplified, rules)

    def apply_rule(self, pattern, replacement, recursive=True):
        """Apply a single rules to the expression

        This is equivalent to :meth:`apply_rules` with
        ``rules=[(pattern, replacement)]``

        Args:
            pattern (.Pattern): A pattern containing one or more wildcards
            replacement (callable): A callable that takes the wildcard names in
                `pattern` as keyword arguments, and returns a replacement for
                any expression that `pattern` matches.

        Example:
            Consider the following Heisenberg Hamiltonian::

                >>> tls = SpinSpace(label='s', spin='1/2')
                >>> i, j, n = symbols('i, j, n', cls=IdxSym)
                >>> J = symbols('J', cls=sympy.IndexedBase)
                >>> def Sig(i):
                ...     return OperatorSymbol(
                ...         StrLabel(sympy.Indexed('sigma', i)), hs=tls)
                >>> H = - Sum(i, tls)(Sum(j, tls)(
                ...     J[i, j] * Sig(i) * Sig(j)))
                >>> unicode(H)
                '- (∑_{i,j ∈ ℌₛ} J_ij σ̂_i^(s) σ̂_j^(s))'

            We can transform this into a classical Hamiltonian by replacing the
            operators with scalars::

                >>> H_classical = H.apply_rule(
                ...     pattern(OperatorSymbol, wc('label', head=StrLabel)),
                ...     lambda label: label.expr * IdentityOperator)
                >>> unicode(H_classical)
                '- (∑_{i,j ∈ ℌₛ} J_ij σ_i σ_j)'
        """
        return self.apply_rules([(pattern, replacement)], recursive=recursive)

    def rebuild(self):
        """Recursively re-instantiate the expression

        This is generally used within a managed context such as
        :func:`.extra_rules`, :func:`.extra_binary_rules`, or
        :func:`.no_rules`.
        """
        return self.apply_rules(rules={})

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
        if self._free_symbols is None:
            res = set.union(
                set([]), # dummy arg (union fails without arguments)
                *[_free_symbols(val) for val in self.kwargs.values()])
            res.update(
                set([]), # dummy arg (update fails without arguments)
                *[_free_symbols(arg) for arg in self.args])
            self._free_symbols = res
        return self._free_symbols

    @property
    def bound_symbols(self):
        """Set of bound SymPy symbols in the expression"""
        if self._bound_symbols is None:
            res = set.union(
                set([]), # dummy arg (union fails without arguments)
                *[_bound_symbols(val) for val in self.kwargs.values()])
            res.update(
                set([]), # dummy arg (update fails without arguments)
                *[_bound_symbols(arg) for arg in self.args])
            self._bound_symbols = res
        return self._bound_symbols

    @property
    def all_symbols(self):
        """Combination of :attr:`free_symbols` and :attr:`bound_symbols`"""
        if self._all_symbols is None:
            self._all_symbols = self.free_symbols | self.bound_symbols
        return self._all_symbols

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


def _apply_rules_no_recurse(expr, rules):
    """Non-recursively match expr again all rules"""
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


def _apply_rules(expr, rules):
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
        logger = logging.getLogger('QNET.create')
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
                expr = _apply_rules_no_recurse(expr, rules)
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
                    stack[-1][i] = _apply_rules_no_recurse(arg, rules)
                    if LOG:
                        logger.debug(
                            "   arg is leaf, replacing with simplified expr: "
                            "%s", stack[-1][i])
                    path[-1] += 1
    else:
        return _apply_rules_no_recurse(expr, rules)


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
    """Base class for "operations"

    Operations are Expressions that act algebraically on other expressions
    (their "operands").

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
