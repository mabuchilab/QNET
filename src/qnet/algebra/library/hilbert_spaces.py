"""Specialized :class:`.LocalSpace` subclasses for physical systems"""

import sympy

from ..core.hilbert_space_algebra import LocalSpace
from ...utils.indices import FockIndex, SpinIndex

__all__ = ['FockSpace', 'SpinSpace']


class FockSpace(LocalSpace):
    """A Hilbert space whose basis states correspond to an "excitation" (in the
    most abstract sense). A Fock space can be infinite, with levels labeled by
    integers 0, 1, ...::

        >>> hs = FockSpace(label=0)

    or truncated to a finite dimension::

        >>> hs = FockSpace(0, dimension=5)
        >>> hs.basis_labels
        ('0', '1', '2', '3', '4')

    For finite-dimensional (truncated) Fock spaces, we also allow an arbitrary
    alternative labeling of the canonical basis::

        >>> hs = FockSpace('rydberg', dimension=3, basis=('g', 'e', 'r'))

    A :class:`.BasisKet` associated with a :class:`FockSpace` may use a
    :class:`.FockIndex` symbolic label.
    """
    _basis_label_types = (int, str, FockIndex)  # acceptable types for labels


class SpinSpace(LocalSpace):
    """A Hilbert space for an integer or half-integer spin system

    For a given spin $N$, the resulting Hilbert space has dimension $2 N + 1$
    with levels labeled from $-N$ to $+N$ (as strings)

    For an integer spin::

        >>> hs = SpinSpace(label=0, spin=1)
        >>> hs.dimension
        3
        >>> hs.basis_labels
        ('-1', '0', '+1')

    For a half-integer  spin::

        >>> hs = SpinSpace(label=0, spin=sympy.Rational(3, 2))
        >>> hs.dimension
        4
        >>> hs.basis_labels
        ('-3/2', '-1/2', '+1/2', '+3/2')

    Raises:

        ValueError: if `spin` is not an integer or half-integer greater than
            zero
    """
    _basis_label_types = (str, FockIndex)  # acceptable types for labels

    def __init__(
            self, label, *, spin, local_identifiers=None, order_index=None):
        spin = sympy.sympify(spin)
        if not (2 * spin).is_integer:
            raise ValueError(
                "spin %s must be an integer or half-integer" % spin)
        try:
            dimension = int(2 * spin) + 1
        except TypeError:
            raise ValueError(
                "spin %s must be an integer or half-integer" % spin)
        if dimension <= 1:
            raise ValueError("spin %s must be greater than zero" % spin)
        bottom = -spin
        basis = [
            SpinIndex(bottom + n).evaluate({}) for n in range(dimension)]
        super().__init__(
            label=label, basis=basis, dimension=dimension,
            local_identifiers=local_identifiers, order_index=order_index)
