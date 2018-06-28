from contextlib import contextmanager
from collections import OrderedDict
from copy import copy

from ..core.abstract_algebra import Expression
from ...utils.check_rules import check_rules_dict


__all__ = [
    "no_instance_caching", "temporary_instance_cache", "extra_rules",
    "extra_binary_rules", "no_rules"]


@contextmanager
def no_instance_caching():
    """Temporarily disable the caching of instances through
    :meth:`.Expression.create`
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
    rules = check_rules_dict(rules)
    cls._rules.update(check_rules_dict(rules))
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
    cls._binary_rules.update(check_rules_dict(rules))
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
        cls._rules = OrderedDict([])
    except AttributeError:
        has_rules = False
    try:
        orig_binary_rules = cls._binary_rules
        cls._binary_rules = OrderedDict([])
    except AttributeError:
        has_binary_rules = False
    yield
    if has_rules:
        cls._rules = orig_rules
    if has_binary_rules:
        cls._binary_rules = orig_binary_rules
    cls._instances = orig_instances
