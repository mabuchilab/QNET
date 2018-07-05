from contextlib import contextmanager
from collections import OrderedDict

from ..core.abstract_algebra import Expression


__all__ = [
    "no_instance_caching", "temporary_instance_cache", "temporary_rules"]


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
    yield
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
    yield
    for i, cls in enumerate(classes):
        cls._instances = orig_instances[i]


@contextmanager
def temporary_rules(*classes, clear=False):
    """Allow temporary modification of rules for :meth:`~.Expression.create`

    For every one of the given `classes`, temporarily disable all rules
    (processed by :func:`.match_replace` or :func:`.match_replace_binary`).
    Implies :func:`temporary_instance_cache`. If `clear` is given as True, all
    existing rules are temporarily cleared from the given classes.

    Within the managed context, :meth:`~.Expression.add_rule` may be used for
    any class in `classes` to define local rules, or
    :meth:`~.Expression.del_rules` to disable specific existing rules (assuming
    `clear` is False). Upon leaving the managed context all original rules will
    be restored, removing any local rules.

    The classes' :attr:`~.Expression.simplifications` attribute is also
    protected from permanent modification. Locally modifying
    :attr:`~.Expression.simplifications` should be done with care, but allows
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

    yield

    for i, cls in enumerate(classes):
        cls._instances = orig_instances[i]
        cls.simplifications = orig_simplifications[i]
        if orig_rules[i] is not None:
            cls._rules = orig_rules[i]
        if orig_binary_rules[i] is not None:
            cls._binary_rules = orig_binary_rules[i]
