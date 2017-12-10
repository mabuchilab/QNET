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

"""Collection of tools to manually manipulate algebraic expressions"""

from functools import partial
from collections import OrderedDict

from .abstract_algebra import simplify
from .operator_algebra import Operator, Commutator, OperatorTimes
from .pattern_matching import pattern, wc

__all__ = [
    'expand_commutators_leibniz', 'evaluate_commutators', 'simplify_by_method',
    'expand_indexed_sum']

__private__ = []  # anything not in __all__ must be in __private__


def expand_commutators_leibniz(expr, expand_expr=True):
    """Recursively expand commutators in `expr` according to the Leibniz rule.

    .. math::

        [A B, C] = A [B, C] + [A, C] B

    .. math::

        [A, B C] = [A, B] C + B [A, C]

    If `expand_expr` is True, expand products of sums in `expr`, as well as in
    the result.
    """
    recurse = partial(expand_commutators_leibniz, expand_expr=expand_expr)
    A = wc('A', head=Operator)
    C = wc('C', head=Operator)
    AB = wc('AB', head=OperatorTimes)
    BC = wc('BC', head=OperatorTimes)

    def leibniz_right(A, BC):
        """[A, BC] -> [A, B] C + B [A, C]"""
        B = BC.operands[0]
        C = OperatorTimes(*BC.operands[1:])
        return Commutator.create(A, B) * C + B * Commutator.create(A, C)

    def leibniz_left(AB, C):
        """[AB, C] -> A [B, C] C + [A, C] B"""
        A = AB.operands[0]
        B = OperatorTimes(*AB.operands[1:])
        return A * Commutator.create(B, C) + Commutator.create(A, C) * B

    rules = OrderedDict([
        ('leibniz1', (
            pattern(Commutator, A, BC),
            lambda A, BC: recurse(leibniz_right(A, BC).expand()))),
        ('leibniz2', (
            pattern(Commutator, AB, C),
            lambda AB, C: recurse(leibniz_left(AB, C).expand())))])

    if expand_expr:
        res = simplify(expr.expand(), rules).expand()
    else:
        res = simplify(expr, rules)
    return res


def evaluate_commutators(expr):
    """Evaluate all commutators in `expr`.

    All commutators are evaluated as the explicit formula

    .. math::

        [A, B] = A B - B A

    """
    A = wc('A', head=Operator)
    B = wc('B', head=Operator)
    return simplify(
        expr, [(pattern(Commutator, A, B), lambda A, B: A*B - B*A)])


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


def expand_indexed_sum(expr, max_terms=None):
    """Expand indexed sums by calling the `expand_sum` method on any
    sub-expression. Truncate after `max_terms`."""
    return simplify_by_method(expr, 'expand_sum', max_terms=max_terms)
