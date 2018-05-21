r"""
The `ordering` package implements the default canonical ordering for sums and
products of operators, states, and superoperators.

To the extent that commutativity rules allow this, the ordering defined here
groups objects of the same Hilbert space together, and orders these groups in
the same order that the Hilbert spaces occur in a `ProductSpace`
(lexicographically/by `order_index`/by complexity). Objects within the same
Hilbert space (again, assuming they commute) are ordered by the `KeyTuple`
value that `expr_order_key` returns for each object. Note that `expr_order_key`
defers to the object's `_order_key` property, if available. This property
should be defined for all QNET Expressions, generally ordering objects
according to their type, then their label (if any), then their pre-factor then
any other properties.

We assume that quantum operations have either full commutativity (sums, or
products of states), or commutativity of objects only in different Hilbert
spaces (e.g. products of operators). The former is handled by
`FullCommutativeHSOrder`, the latter by `DisjunctCommutativeHSOrder`. Theses
classes serve as the `order_key` for sums and products (e.g. `OperatorPlus` and
similar classes)

A user may implement a custom ordering by subclassing (or replacing)
`FullCommutativeHSOrder` and/or `DisjunctCommutativeHSOrder`, and assigning
their replacements to all the desired algebraic classes.
"""

from collections import OrderedDict

__all__ = []

__private__ = [  # anything not in __all__ must be in __private__
    'KeyTuple', 'expr_order_key', 'DisjunctCommutativeHSOrder',
    'FullCommutativeHSOrder'
]


class KeyTuple(tuple):
    """A tuple that allows for ordering, facilitating the default ordering of
    Operations. It differs from a normal tuple in that it falls back to string
    comparison if any elements are not directly comparable"""
    def __lt__(self, other):
        for (a, b) in zip(self, other):
            try:
                if a < b:
                    return True
                elif a > b:
                    return False
            except (TypeError, ValueError):
                if str(a) < str(b):
                    return True
                elif str(a) > str(b):
                    return False
        if len(self) < len(other):
            return True
        elif len(self) > len(other):
            return False
        return None

    def __repr__(self):
        return self.__class__.__name__ + tuple.__repr__(self)


def expr_order_key(expr):
    """A default order key for arbitrary expressions"""
    if hasattr(expr, '_order_key'):
        return expr._order_key
    try:
        if isinstance(expr.kwargs, OrderedDict):
            key_vals = expr.kwargs.values()
        else:
            key_vals = [expr.kwargs[key] for key in sorted(expr.kwargs)]
        return KeyTuple((expr.__class__.__name__, ) +
                        tuple(map(expr_order_key, expr.args)) +
                        tuple(map(expr_order_key, key_vals)))
    except AttributeError:
        return str(expr)


class DisjunctCommutativeHSOrder():
    """Auxiliary class that generates the correct pseudo-order relation for
    operator products.  Only operators acting on disjoint Hilbert spaces
    are commuted to reflect the order the local factors have in the total
    Hilbert space. I.e., ``sorted(factors, key=DisjunctCommutativeHSOrder)``
    achieves this ordering.
    """

    def __init__(self, op, space_order=None, op_order=None):
        from qnet.algebra.core.hilbert_space_algebra import TrivialSpace
        self.op = op
        self.space = op.space
        if space_order is None:
            self._space_order = op.space._order_key
        else:
            self._space_order = space_order
        if op_order is None:
            self._op_order = expr_order_key(op)
        else:
            self._op_order = op_order
        self.trivial = False
        if op.space is TrivialSpace:
            self.trivial = True

    def __repr__(self):
        from qnet.printing import srepr
        return "%s(%s, space_order=%r, op_order=%r)" % (
                self.__class__.__name__, srepr(self.op), self._space_order,
                self._op_order)

    def __lt__(self, other):
        if self.trivial and other.trivial:
            return self._op_order < other._op_order
        else:
            if self.space.isdisjoint(other.space):
                return self._space_order < other._space_order
        return None  # no ordering


class FullCommutativeHSOrder():
    """Auxiliary class that generates the correct pseudo-order relation for
    operator sums.  Operators are first ordered by their Hilbert space, then by
    their order-key; ``sorted(factors, key=FullCommutativeHSOrder)``
    achieves this ordering.
    """

    def __init__(self, op, space_order=None, op_order=None):
        self.op = op
        self.space = op.space
        if space_order is None:
            self._space_order = self.space._order_key
        else:
            self._space_order = space_order
        if op_order is None:
            self._op_order = expr_order_key(op)
        else:
            self._op_order = op_order

    def __repr__(self):
        from qnet.printing import srepr
        return "%s(%s, space_order=%r, op_order=%r)" % (
                self.__class__.__name__, srepr(self.op), self._space_order,
                self._op_order)

    def __lt__(self, other):
        if self.space == other.space:
            return self._op_order < other._op_order
        else:
            return self._space_order < other._space_order
