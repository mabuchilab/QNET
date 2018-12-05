import sympy
from contextlib import contextmanager
from collections import OrderedDict

from ..core.abstract_algebra import Expression


__all__ = [
    "no_instance_caching", "temporary_instance_cache", "temporary_rules",
    "symbols"]


@contextmanager
def no_instance_caching():
    """Temporarily disable instance caching in :meth:`~.Expression.create`

    Within the managed context, :meth:`~.Expression.create` will not use any
    caching, for any class.
    """
    # this assumes that no sub-class of Expression shadows
    # Expression.instance_caching
    orig_flag = Expression.instance_caching
    Expression.instance_caching = False
    try:
        yield
    finally:
        Expression.instance_caching = orig_flag


@contextmanager
def temporary_instance_cache(*classes):
    """Use a temporary cache for instances in :meth:`~.Expression.create`

    The instance cache used by :meth:`~.Expression.create` for any of the given
    `classes` will be cleared upon entering the managed context, and restored
    on leaving it.  That is, no cached instances from outside of the managed
    context will be used within the managed context, and vice versa
    """
    orig_instances = []
    for cls in classes:
        orig_instances.append(cls._instances)
        cls._instances = {}
    try:
        yield
    finally:
        for i, cls in enumerate(classes):
            cls._instances = orig_instances[i]


@contextmanager
def temporary_rules(*classes, clear=False):
    """Allow temporary modification of rules for :meth:`~.Expression.create`

    For every one of the given `classes`, protect the rules (processed by
    :func:`.match_replace` or :func:`.match_replace_binary`) associated with
    that class from modification beyond the managed context.
    Implies :func:`temporary_instance_cache`. If `clear` is given as
    True, all existing rules are temporarily cleared from the given classes on
    entering the managed context.

    Within the managed context, :meth:`~.Expression.add_rule` may be used for
    any class in `classes` to define local rules, or
    :meth:`~.Expression.del_rules` to disable specific existing rules (assuming
    `clear` is False). Upon leaving the managed context all original rules will
    be restored, removing any local rules.

    The `classes`' :obj:`simplifications <.Expression>` attribute is also
    protected from permanent modification. Locally modifying
    :obj:`simplifications <.Expression>` should be done with care, but allows
    complete control over the creation of expressions.
    """
    orig_instances = []
    orig_rules = []
    orig_binary_rules = []
    orig_simplifications = []

    for cls in classes:
        orig_instances.append(cls._instances)
        cls._instances = {}
        orig_simplifications.append(cls.simplifications)
        cls.simplifications = cls.simplifications.copy()
        try:
            orig_rules.append(cls._rules)
            if clear:
                cls._rules = OrderedDict([])
            else:
                cls._rules = cls._rules.copy()
        except AttributeError:
            orig_rules.append(None)
        try:
            orig_binary_rules.append(cls._binary_rules)
            if clear:
                cls._binary_rules = OrderedDict([])
            else:
                cls._binary_rules = cls._binary_rules.copy()
        except AttributeError:
            orig_binary_rules.append(None)

    try:
        yield
    finally:
        for i, cls in enumerate(classes):
            cls._instances = orig_instances[i]
            cls.simplifications = orig_simplifications[i]
            if orig_rules[i] is not None:
                cls._rules = orig_rules[i]
            if orig_binary_rules[i] is not None:
                cls._binary_rules = orig_binary_rules[i]


def symbols(names, **args):
    """The :func:`~sympy.core.symbol.symbols` function from SymPy

    This can be used to generate QNET symbols as well::

        >>> A, B, C = symbols('A B C', cls=OperatorSymbol, hs=0)
        >>> srepr(A)
        "OperatorSymbol('A', hs=LocalSpace('0'))"
        >>> C1, C2 = symbols('C_1:3', cls=CircuitSymbol, cdim=2)
        >>> srepr(C1)
        "CircuitSymbol('C_1', cdim=2)"

    Basically, the `cls` keyword argument can be any instantiator, i.e. a class
    or callable that receives a symbol name as the single positional argument.
    Any keyword arguments not handled by :func:`symbols` directly (see
    :func:`sympy.core.symbol.symbols` documentation) is passed on to the
    instantiator. Obviously, this is extremely flexible.

    Note:
        :func:`symbol` does not pass *positional* arguments to the
        instantiator. Two possible workarounds to create symbols with e.g. a
        scalar argument are::

            >>> t = symbols('t', positive=True)
            >>> A_t, B_t = symbols(
            ...     'A B', cls=lambda s: OperatorSymbol(s, t, hs=0))
            >>> srepr(A_t, cache={t: 't'})
            "OperatorSymbol('A', t, hs=LocalSpace('0'))"
            >>> A_t, B_t = (OperatorSymbol(s, t, hs=0) for s in ('A', 'B'))
            >>> srepr(B_t, cache={t: 't'})
            "OperatorSymbol('B', t, hs=LocalSpace('0'))"
    """
    # this wraps the sympy symbols function (instead of just importing and
    # exposing it directly) solely for the extra documentation
    return sympy.symbols(names, **args)
