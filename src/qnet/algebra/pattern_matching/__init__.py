r'''QNET's pattern matching engine.

Patterns may be constructed by either instantiating a :class:`Pattern` instance
directly, or (preferred) by calling the :func:`pattern`, :func:`pattern_head`,
or :func:`wc` helper routines.

The pattern may then be matched against an expression using
:func:`match_pattern`. The result of a match is a :class:`MatchDict` object,
which evaluates to True or False in a boolean context to indicate the success
or failure of the match (or alternatively, through the `success` attribute).
The :class:`MatchDict` object also maps any wildcard names to the expression
that the corresponding wildcard Pattern matches.
'''
import re
import unittest.mock
from collections import OrderedDict
from collections.abc import Sequence


__all__ = [
    'MatchDict', 'Pattern', 'match_pattern', 'pattern', 'pattern_head', 'wc']

__private__ = ['ProtoExpr']  # anything not in __all__ must be in __private__


class MatchDict(OrderedDict):
    """Result of a :meth:`Pattern.match`

    Dictionary of wildcard names to expressions. Once the value for a key is
    set, attempting to set it again with a different value raises a
    :exc:`KeyError`.  The attribute `merge_lists` may be set to modify this
    behavior for values that are lists: If it is set to a value different from
    zero, two lists that are set via the same key are merged. If `merge_lists`
    is negative, the new values are appended to the existing values; if it is
    positive, the new values are prepended.

    In a boolean context, a :class:`MatchDict` always evaluates as True (even
    if empty, unlike a normal dictionary), unless the `success` attribute is
    explicitly set to False (which a failed :meth:`Pattern.match` should
    do)

    Attributes:
        success (bool):  Value of the :class:`MatchDict` object in a boolean
            context: ``bool(match) == match.success``
        reason (str):  If `success` is False, string explaining why the match
            failed
        merge_lists (int): Code that indicates how to combine multiple values
            that are lists
    """

    def __init__(self, *args):
        self.success = True
        self.reason = ""
        self._len = 0
        self.merge_lists = 0
        super().__init__(*args)

    def __delitem__(self, key, **kwargs):
        raise KeyError('Read-only dictionary')

    def __setitem__(self, key, value, **kwargs):
        if key in self:
            if isinstance(self[key], list) and isinstance(value, list):
                if self.merge_lists < 0:
                    self[key].extend(value)
                    return
                elif self.merge_lists > 0:
                    self[key][0:0] = value
                    return
            if self[key] == value:
                return
            else:
                raise KeyError('{} has already been set'.format(key))
        self._len += 1
        OrderedDict.__setitem__(self, key, value)

    def __len__(self):
        return self._len

    def __bool__(self):
        return self.success

    def update(self, *others):
        """Update dict with entries from `other`

        If `other` has an attribute ``success=False`` and ``reason``, those
        attributes are copied as well
        """
        for other in others:
            for key, val in other.items():
                self[key] = val
            try:
                if not other.success:
                    self.success = False
                    self.reason = other.reason
            except AttributeError:
                pass

    __non_zero__ = __bool__


class Pattern():
    """Pattern for matching an expression

    Args:
        head (type or None):  The type (or tuple of types) of the expression
            that can be matched. If None, any type of Expression
            matches
        args (list or None): List or tuple of positional arguments of the
            matched Expression (cf. `Expression.args`). Each element is an
            expression (to be matched exactly) or another Pattern instance
            (matched recursively). If None, no arguments are checked
        kwargs (dict or None): Dictionary of keyword arguments of the
            expression (cf. `Expression.kwargs`). As for `args`, each value is
            an expression or Pattern instance.
        mode (int): If the pattern is used to match the arguments of an
            expression, code to indicate how many arguments the Pattern can
            consume: `Pattern.single`, `Pattern.one_or_more`,
            `Pattern.zero_or_more`
        wc_name (str or None): If pattern matches an expression, key in the
            resulting :class:`MatchDict` for the expression. If None, the match
            will not be recorded in the result
        conditions (list of callables, or None): If not None, a list of
            callables that take `expr` and return a boolean value. If the
            return value is False, the pattern is determined not to match
            `expr`.

    Note:
        For (sub-)patterns that occur nested in the `args` attribute of another
        pattern, only the first or last sub-pattern may have a `mode` other
        than `Pattern.single`. This also implies that only one of the `args`
        may have a `mode` other than `Pattern.single`. This restrictions
        ensures that patterns can be matched without backtracking, thus
        guaranteeing numerical efficiency.

    Example:

        Consider the following nested circuit expression::

            >>> C1 = CircuitSymbol('C1', cdim=3)
            >>> C2 = CircuitSymbol('C2', cdim=3)
            >>> C3 = CircuitSymbol('C3', cdim=3)
            >>> C4 = CircuitSymbol('C4', cdim=3)
            >>> perm1 = CPermutation((2, 1, 0))
            >>> perm2 = CPermutation((0, 2, 1))
            >>> concat_expr = Concatenation(
            ...                   (C1 << C2 << perm1),
            ...                   (C3 << C4 << perm2))

        We may match this with the following pattern::

            >>> conditions = [lambda c: c.cdim == 3,
            ...               lambda c: c.label[0] == 'C']
            >>> A__Circuit = wc("A__", head=CircuitSymbol,
            ...                 conditions=conditions)
            >>> C__Circuit = wc("C__", head=CircuitSymbol,
            ...                 conditions=conditions)
            >>> B_CPermutation = wc("B", head=CPermutation)
            >>> D_CPermutation = wc("D", head=CPermutation)
            >>> pattern_concat = pattern(
            ...         Concatenation,
            ...         pattern(SeriesProduct, A__Circuit, B_CPermutation),
            ...         pattern(SeriesProduct, C__Circuit, D_CPermutation))
            >>> m = pattern_concat.match(concat_expr)

        The match returns the following dictionary::

            >>> result = {'A': [C1, C2], 'B': perm1, 'C': [C3, C4], 'D': perm2}
            >>> assert m == result
    """
    # Note: if we ever need to allow Patterns that have backtracking (i.e.
    # multiple more-than-single wildcards, or more-than-single wildcards
    # sandwiched between single-wildcards, we should subclass Pattern to
    # BacktrackingPattern(Pattern) to implement this special case

    single = 1
    one_or_more = 2
    zero_or_more = 3

    def __init__(self, head=None, args=None, kwargs=None, *, mode=1,
                 wc_name=None, conditions=None):
        self._str = None
        self._repr = None
        if head is not None:
            if not hasattr(head, '__name__'):
                for sub_head in head:
                    if not hasattr(sub_head, '__name__'):
                        # during doc generation, some types are mocked and are
                        # missing the __name__ attribute
                        if not isinstance(sub_head, unittest.mock.Mock):
                            raise TypeError("'head' must be class or tuple of "
                                            "classes")
        self.head = head
        if args is not None:
            args = tuple(args)
        self.args = args
        self._has_non_single_arg = False  # args contains Pattern with mode>1?
        self._non_single_arg_on_left = False   # is that Pattern in args[0]?
        if args is not None:
            for i, arg in enumerate(args):
                if isinstance(arg, Pattern):
                    if arg.mode > self.single:
                        self._has_non_single_arg = True
                        if i == 0:
                            self._non_single_arg_on_left = True
                        elif i < len(args) - 1:
                            raise ValueError(
                                    "Only the first or last argument may have "
                                    "a mode indicating an occurrence of more "
                                    "than 1")
        self.kwargs = kwargs
        self.mode = mode
        if mode not in [self.single, self.one_or_more, self.zero_or_more]:
            raise ValueError(("Mode must be one of {cls}.single, "
                              "{cls}.one_or_more, or {cls}.zero_or_more")
                             .format(cls=str(self.__class__.__name__)))
        self.wc_name = wc_name
        if conditions is None:
            self.conditions = []
        else:
            self.conditions = conditions
        self._repr = None  # lazy evaluation
        self._arg_iterator = iter
        if self._non_single_arg_on_left:
            # When the non-single argument is on the left, we move through
            # self.args and the args of any matched expression in reverse (such
            # that the args that may match an indefinite number of times are
            # the last ones to be considered). This goes together with setting
            # res.merge_lists = 1 in the `match` method
            self._arg_iterator = reversed

    def extended_arg_patterns(self):
        """Iterator over patterns for positional arguments to be matched

        This yields the elements of :attr:`args`, extended by their `mode`
        value
        """
        for arg in self._arg_iterator(self.args):
            if isinstance(arg, Pattern):
                if arg.mode > self.single:
                    while True:
                        yield arg
                else:
                    yield arg
            else:
                yield arg

    def _check_last_arg_pattern(self, current_arg_pattern, last_arg_pattern):
        """Given a "current" arg pattern (that was used to match the last
        actual argument of an expression), and another ("last") argument
        pattern, raise a ValueError, unless the "last" argument pattern is a
        "zero or more" wildcard. In that case, return a dict that maps the
        wildcard name to an empty list
        """
        try:
            if last_arg_pattern.mode == self.single:
                raise ValueError("insufficient number of arguments")
            elif last_arg_pattern.mode == self.zero_or_more:
                if last_arg_pattern.wc_name is not None:
                    if last_arg_pattern != current_arg_pattern:
                        # we have to record an empty match
                        return {last_arg_pattern.wc_name: []}
            elif last_arg_pattern.mode == self.one_or_more:
                if last_arg_pattern != current_arg_pattern:
                    raise ValueError("insufficient number of arguments")
        except AttributeError:
            raise ValueError("insufficient number of arguments")
        return {}

    def match(self, expr) -> MatchDict:
        """Match the given expression (recursively)

        Returns a :class:`MatchDict` instance that maps any wildcard names to
        the expressions that the corresponding wildcard pattern matches. For
        (sub-)pattern that have a `mode` attribute other than `Pattern.single`,
        the wildcard name is mapped to a list of all matched expression.

        If the match is successful, the resulting :class:`MatchDict` instance
        will evaluate to True in a boolean context. If the match is not
        successful, it will evaluate as False, and the reason for failure is
        available in the `reason` attribute of the :class:`MatchDict` object.
        """
        res = MatchDict()
        if self._has_non_single_arg:
            if self._non_single_arg_on_left:
                res.merge_lists = 1
            else:
                res.merge_lists = -1
        if self.head is not None:
            if not isinstance(expr, self.head):
                res.reason = ("%s is not an instance of %s"
                              % (repr(expr), self._repr_head()))
                res.success = False
                return res
        for i_cond, condition in enumerate(self.conditions):
            if not condition(expr):
                res.reason = ("%s does not meet condition %d"
                              % (repr(expr), i_cond+1))
                res.success = False
                return res
        try:
            if self.args is not None:
                arg_pattern = self.extended_arg_patterns()
                for arg in self._arg_iterator(expr.args):
                    current_arg_pattern = next(arg_pattern)
                    res.update(match_pattern(current_arg_pattern, arg))
                    if not res.success:
                        return res
                # ensure that we have matched all arg patterns
                try:
                    last_arg_pattern = next(arg_pattern)
                    res.update(self._check_last_arg_pattern(
                                    current_arg_pattern, last_arg_pattern))
                except StopIteration:
                    pass  # expected, if current_arg_pattern was the last one
            if self.kwargs is not None:
                for key, arg_pattern in self.kwargs.items():
                    res.update(match_pattern(arg_pattern, expr.kwargs[key]))
                    if not res.success:
                        return res
        except AttributeError as exc_info:
            res.reason = ("%s is a scalar, not an Expression: %s"
                          % (repr(expr), str(exc_info)))
            res.success = False
        except ValueError as exc_info:
            res.reason = "%s: %s" % (repr(expr), str(exc_info))
            res.success = False
        except StopIteration:
            res.reason = ("%s has an too many positional arguments"
                          % repr(expr))
            res.success = False
        except KeyError as exc_info:
            if "has already been set" in str(exc_info):
                res.reason = "Double wildcard: %s" % str(exc_info)
            else:
                res.reason = ("%s has no keyword argument %s"
                              % (repr(expr), str(exc_info)))
            res.success = False
        if res.success:
            if self.wc_name is not None:
                try:
                    if self.mode > self.single:
                        res[self.wc_name] = [expr, ]
                    else:
                        res[self.wc_name] = expr
                except KeyError as exc_info:
                    res.reason = "Double wildcard: %s" % str(exc_info)
                    res.success = False
        return res

    def findall(self, expr):
        """list of all matching (sub-)expressions in `expr`

        See also:
            :meth:`finditer` yields the matches (:class:`MatchDict` instances)
            for the matched expressions.
        """
        result = []
        try:
            for arg in expr.args:
                result.extend(self.findall(arg))
            for arg in expr.kwargs.values():
                result.extend(self.findall(arg))
        except AttributeError:
            pass
        if self.match(expr):
            result.append(expr)
        return result

    def finditer(self, expr):
        """Return an iterator over all matches in `expr`

        Iterate over all :class:`MatchDict` results of matches for any
        matching (sub-)expressions in `expr`. The order of the matches conforms
        to the equivalent matched expressions returned by :meth:`findall`.
        """
        try:
            for arg in expr.args:
                for m in self.finditer(arg):
                    yield m
            for arg in expr.kwargs.values():
                for m in self.finditer(arg):
                    yield m
        except AttributeError:
            pass
        m = self.match(expr)
        if m:
            yield m

    @property
    def wc_names(self):
        """Set of all wildcard names occurring in the pattern"""
        if self.wc_name is None:
            res = set()
        else:
            res = set([self.wc_name])
        if self.args is not None:
            for arg in self.args:
                if isinstance(arg, Pattern):
                    res.update(arg.wc_names)
        if self.kwargs is not None:
            for val in self.kwargs.values():
                if isinstance(val, Pattern):
                    res.update(val.wc_names)
        return res

    def _repr_head(self):
        if self.head is not None:
            if isinstance(self.head, (list, tuple)):
                heads = [head.__name__ for head in self.head]
                return '[' + ", ".join(heads) + ']'
            else:
                return self.head.__name__

    def __repr__(self):
        if self._repr is None:
            cls = str(self.__class__.__name__)
            mode_str = {
                self.single: cls + '.single',
                self.one_or_more: cls + '.one_or_more',
                self.zero_or_more: cls + '.zero_or_more',
            }
            res = []
            if self.head is not None:
                res.append('head=' + self._repr_head())
            if self.args is not None:
                res.append('args=' + repr(self.args))
            if self.kwargs is not None:
                key_vals = []
                for key in sorted(self.kwargs):
                    key_vals.append("'%s': %s" % (key, self.kwargs[key]))
                res.append('kwargs={' + ", ".join(key_vals) + "}")
            if self.mode != 1:
                res.append('mode=' + mode_str[self.mode])
            if self.wc_name is not None:
                res.append('wc_name=' + repr(self.wc_name))
            if len(self.conditions) > 0:
                res.append('conditions=' + repr(self.conditions))
            self._repr = cls + "(" + ", ".join(res) + ")"
        return self._repr

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return ((self.__class__ == other.__class__) and
                all([(getattr(self, attr) == getattr(other, attr))
                     for attr in ('args', 'kwargs', 'mode', 'wc_name',
                                  'conditions', )]))


def pattern(head, *args, mode=1, wc_name=None, conditions=None, **kwargs) \
        -> Pattern:
    """'Flat' constructor for the Pattern class

    Positional and keyword arguments are mapped into `args` and `kwargs`,
    respectively. Useful for defining rules that match an instantiated
    Expression with specific arguments
    """
    if len(args) == 0:
        args = None
    if len(kwargs) == 0:
        kwargs = None
    return Pattern(head, args, kwargs, mode=mode, wc_name=wc_name,
                   conditions=conditions)


def pattern_head(*args, conditions=None, wc_name=None, **kwargs) -> Pattern:
    """Constructor for a :class:`Pattern` matching a :class:`ProtoExpr`

    The patterns associated with :attr:`_rules` and :attr:`_binary_rules`
    of an :class:`Expression` subclass, or those passed to
    :meth:`Expression.add_rule`, must be instantiated through this
    routine. The function does not allow to set a wildcard name
    (`wc_name` must not be given / be None)"""
    # This routine is indented for the _rules and _binary_rules class
    # attributes of algebraic objects, which the match_replace and
    # match_replace_binary match against a ProtoExpr
    if len(args) == 0:
        args = None
    if len(kwargs) == 0:
        kwargs = None
    if wc_name is not None:
        raise ValueError("pattern_head cannot be used to set a wildcard "
                         "(`wc_name` must not be given")
    pat = Pattern(head=ProtoExpr, args=args, kwargs=kwargs, wc_name=None,
                  conditions=conditions)
    return pat


def wc(name_mode="_", head=None, args=None, kwargs=None, *, conditions=None) \
        -> Pattern:
    """Constructor for a wildcard-:class:`Pattern`

    Helper function to create a Pattern object with an emphasis on wildcard
    patterns, if we don't care about the arguments of the matched expressions
    (otherwise, use :func:`pattern`)

    Args:
        name_mode (str): Combined `wc_name` and `mode` for :class:`Pattern`
            constructor argument. See below for syntax
        head (type, or None): See :class:`Pattern`
        args (list or None): See :class:`Pattern`
        kwargs (dict or None): See :class:`Pattern`
        conditions (list or None): See :class:`Pattern`

    The `name_mode` argument uses trailing underscored to indicate the `mode`:

        * ``A`` -> ``Pattern(wc_name="A", mode=Pattern.single, ...)``
        * ``A_`` -> ``Pattern(wc_name="A", mode=Pattern.single, ...)``
        * ``B__`` -> ``Pattern(wc_name="B", mode=Pattern.one_or_more, ...)``
        * ``B___`` -> ``Pattern(wc_name="C", mode=Pattern.zero_or_more, ...)``
    """
    rx = re.compile(r"^([A-Za-z]?[A-Za-z0-9]*)(_{0,3})$")
    m = rx.match(name_mode)
    if not m:
        raise ValueError("Invalid name_mode: %s" % name_mode)
    wc_name, mode_underscores = m.groups()
    if wc_name == '':
        wc_name = None
    mode = len(mode_underscores) or Pattern.single
    return Pattern(head, args, kwargs, mode=mode, wc_name=wc_name,
                   conditions=conditions)


# In order to match Expressions before they are instantiated, we define
# Proto-Expressions the provide just the 'args' and 'kwargs' properties,
# allowing `match_pattern` to match them via duck typing
class ProtoExpr(Sequence):
    """Object representing an un-instantiated :class:`Expression`

    A :class:`ProtoExpr` may be matched by a :class:`Pattern` created via
    :func:`pattern_head`. This is used in :meth:`.Expression.create`: before an
    expression is instantiated, a :class:`ProtoExpr` is constructed with the
    positional and keyword arguments passed to :meth:`~.Expression.create`.
    Then, this :class:`ProtoExpr` is matched against all the automatic rules
    :meth:`~.Expression.create` knows about.

    Args:
        args (list): positional arguments that would be used in the
            instantiation of the Expression
        kwargs (dict):  keyword arguments. Will we converted to an
            :class:`~.collections.OrderedDict`
        cls (class or None): The class of the Expression that will ultimately
            be instantiated.

    The combined values of `args` and `kwargs` are accessible as a (mutable)
    sequence.
    """
    def __init__(self, args, kwargs, cls=None):
        self.args = list(args)
        self.kwargs = OrderedDict(kwargs)
        self.cls = cls

    def __len__(self):
        return len(self.args) + len(self.kwargs)

    def __getitem__(self, i):
        n_args = len(self.args)
        if i < n_args:
            return self.args[i]
        else:
            return list(self.kwargs.values())[i-n_args]

    def __setitem__(self, i, val):
        n_args = len(self.args)
        if i < n_args:
            self.args[i] = val
        else:
            key = list(self.kwargs.keys())[i-n_args]
            self.kwargs[key] = val

    def __str__(self):
        if self.cls is None:
            cls = 'None'
        else:
            cls = self.cls.__name__
        return "%s(args=%s, kwargs=%s, cls=%s)" % (
                self.__class__.__name__, self.args, self.kwargs, cls)

    def __repr__(self):
        if self.cls is None:
            cls = 'None'
        else:
            cls = self.cls.__name__
        return "%s(args=%r, kwargs=%r, cls=%s)" % (
                self.__class__.__name__, self.args, self.kwargs, cls)

    def instantiate(self, cls=None):
        """Return an instantiated Expression as
        ``cls.create(*self.args, **self.kwargs)``

        Args:
            cls (class): The class of the instantiated expression. If not
            given, ``self.cls`` will be used.
        """
        if cls is None:
            cls = self.cls
        if cls is None:
            raise TypeError("cls must a class")
        return cls.create(*self.args, **self.kwargs)

    @classmethod
    def from_expr(cls, expr):
        """Instantiate proto-expression from the given Expression"""
        return cls(expr.args, expr.kwargs, cls=expr.__class__)

    def __hash__(self):
        return hash((self.__class__, ) + tuple(self.args) +
                    tuple(sorted(self.kwargs.items())))


def match_pattern(expr_or_pattern: object, expr: object) -> MatchDict:
    """Recursively match `expr` with the given `expr_or_pattern`

    Args:
        expr_or_pattern: either a direct expression (equal to `expr` for a
            successful match), or an instance of :class:`Pattern`.
        expr: the expression to be matched
    """
    try:  # first try expr_or_pattern as a Pattern
        return expr_or_pattern.match(expr)
    except AttributeError:  # expr_or_pattern is an expr, not a Pattern
        if expr_or_pattern == expr:
            return MatchDict()  # success
        else:
            res = MatchDict()
            res.success = False
            res.reason = "Expressions '%s' and '%s' are not the same" % (
                          repr(expr_or_pattern), repr(expr))
            return res
