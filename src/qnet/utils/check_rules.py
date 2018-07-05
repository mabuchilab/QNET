"""Utilities for algebraic rules"""

import inspect
from collections.__init__ import OrderedDict

__private__ = ['check_rules_dict']


def check_rules_dict(rules):
    """Verify the `rules` that classes may use for the `_rules` or
    `_binary_rules` class attribute.

    Specifically, `rules` must be a
    :class:`~collections.OrderedDict`-compatible object
    (list of key-value tuples, :class:`dict`,
    :class:`~collections.OrderedDict`)
    that maps a rule name (:class:`str`) to a rule. Each rule consists of a
    :class:`~qnet.algebra.pattern_matching.Pattern` and a replaceent callable.
    The Pattern must be set up to match a
    :class:`~qnet.algebra.pattern_matching.ProtoExpr`. That is,
    the Pattern should be constructed through the
    :func:`~qnet.algebra.pattern_matching.pattern_head` routine.

    Raises:
        TypeError: If `rules` is not compatible with
            :class:`~collections.OrderedDict`, the
            keys in `rules` are not strings, or rule is not a tuple of
            (:class:`~qnet.algebra.pattern_matching.Pattern`, `callable`)
        ValueError: If the `head`-attribute of each Pattern is not an instance
            of :class:`~qnet.algebra.pattern_matching.ProtoExpr`, or if there
            are duplicate keys in `rules`

    Returns:
        :class:`~collections.OrderedDict` of rules
    """
    from qnet.algebra.pattern_matching import Pattern, ProtoExpr

    if hasattr(rules, 'items'):
        items = rules.items()  # `rules` is already a dict / OrderedDict
    else:
        items = rules  # `rules` is a list of (key, value) tuples
    keys = set()
    for key_rule in items:
        try:
            key, rule = key_rule
        except ValueError:
            raise TypeError("rules does not contain (key, rule) tuples")
        if not isinstance(key, str):
            raise TypeError("Key '%s' is not a string" % key)
        if key in keys:
            raise ValueError("Duplicate key '%s'" % key)
        else:
            keys.add(key)
        try:
            pat, replacement = rule
        except TypeError:
            raise TypeError(
                "Rule in '%s' is not a (pattern, replacement) tuple" % key)
        if not isinstance(pat, Pattern):
            raise TypeError(
                "Pattern in '%s' is not a Pattern instance" % key)
        if pat.head is not ProtoExpr:
            raise ValueError(
                "Pattern in '%s' does not match a ProtoExpr" % key)
        if not callable(replacement):
            raise ValueError(
                "replacement in '%s' is not callable" % key)
        else:
            arg_names = inspect.signature(replacement).parameters.keys()
            if not arg_names == pat.wc_names:
                raise ValueError(
                    "arguments (%s) of replacement function differ from the "
                    "wildcard names (%s) in pattern" % (
                        ", ".join(sorted(arg_names)),
                        ", ".join(sorted(pat.wc_names))))
    return OrderedDict(rules)
