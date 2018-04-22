from qnet.algebra.core.abstract_algebra import simplify_by_method
from qnet.algebra.core.indexed_operations import IndexedSum


__all__ = ['expand_indexed_sum']


def expand_indexed_sum(expr, indices=None, max_terms=None):
    """Expand indexed sums by calling the `doit` method on any
    sub-expression. Truncate after `max_terms`."""
    return simplify_by_method(
        expr, 'doit', head=IndexedSum, indices=indices, max_terms=max_terms)
