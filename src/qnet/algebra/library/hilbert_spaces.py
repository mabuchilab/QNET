"""Specialized :class:`.LocalSpace` subclasses for physical systems"""
from collections import OrderedDict
import re

import sympy

from ..core.hilbert_space_algebra import LocalSpace
from ..core.state_algebra import BasisKet
from ...utils.indices import FockIndex, SpinIndex

__all__ = ['FockSpace', 'SpinSpace', 'SpinBasisKet']


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
        >>> hs.spin
        3/2
        >>> hs.dimension
        4
        >>> hs.basis_labels
        ('-3/2', '-1/2', '+1/2', '+3/2')

    For convenience, you may also give `spin` as a tuple::

        >>> hs = SpinSpace(label=0, spin=(3, 2))
        >>> assert hs == SpinSpace(label=0, spin=sympy.Rational(3, 2))

    Raises:

        ValueError: if `spin` is not an integer or half-integer greater than
            zero
    """
    _basis_label_types = (str, SpinIndex)  # acceptable types for labels

    def __init__(
            self, label, *, spin, local_identifiers=None, order_index=None):
        if isinstance(spin, tuple):
            spin = sympy.sympify(spin[0]) / spin[1]
        else:
            spin = sympy.sympify(spin)
        self._spin = spin
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
        basis = tuple([
            SpinIndex(bottom + n).evaluate({}) for n in range(dimension)])

        super().__init__(
            label=label, basis=basis, dimension=dimension,
            local_identifiers=local_identifiers, order_index=order_index)

        # rewrite the kwargs from super()
        self._kwargs = OrderedDict([
            ('spin', self._spin),
            ('local_identifiers', self._sorted_local_identifiers),
            ('order_index', self._order_index)])
        self._minimal_kwargs = self._kwargs.copy()
        if local_identifiers is None:
            del self._minimal_kwargs['local_identifiers']
        if order_index is None:
            del self._minimal_kwargs['order_index']

    def next_basis_label_or_index(self, label_or_index, n=1):
        """Given the label or index of a basis state, return the label
        the next basis state.

        More generally, if `n` is given, return the `n`'th next basis state
        label/index; `n` may also be negative to obtain previous basis state
        labels. Returns a :class:`str` label if `label_or_index` is a
        :class:`str` or :class:`int`, or a :class:`SpinIndex` if
        `label_or_index` is a :class:`SpinIndex`.

        Args:
            label_or_index (int or str or SpinIndex): If `int`, the
                zero-based index of a basis state; if `str`, the label of a
                basis state
            n (int): The increment

        Raises:
            IndexError: If going beyond the last or first basis state
            ValueError: If `label` is not a label for any basis state in the
                Hilbert space
            .BasisNotSetError: If the Hilbert space has no defined basis
            TypeError: if `label_or_index` is neither a :class:`str` nor an
                :class:`int`, nor a :class:`SpinIndex`

        Note:
            This differs from its super-method only by never returning an
            integer index (which is not accepted when instantiating a
            :class:`BasisKet` for a :class:`SpinSpace`)
        """
        if isinstance(label_or_index, int):
            new_index = label_or_index + n
            if new_index < 0:
                raise IndexError("index %d < 0" % new_index)
            if new_index >= self.dimension:
                raise IndexError(
                    "index %d out of range for basis %s"
                    % (new_index, self._basis))
            return self.basis_labels[new_index]
        elif isinstance(label_or_index, str):
            label_index = self.basis_labels.index(label_or_index)
            new_index = label_index + n
            if (new_index < 0) or (new_index >= len(self._basis)):
                raise IndexError(
                    "index %d out of range for basis %s"
                    % (new_index, self._basis))
            return self.basis_labels[new_index]
        elif isinstance(label_or_index, SpinIndex):
            return label_or_index.__class__(expr=label_or_index.expr + n)

    @property
    def spin(self) -> sympy.Rational:
        """The spin-number associated with the :class:`SpinSpace`

        This can be a SymPy integer or a half-integer.
        """
        return self._spin

    @property
    def multiplicity(self) -> int:
        """The multiplicity of the Hilbert space, $2 S + 1$.

        This is equivalent to the dimension::

            >>> hs = SpinSpace('s', spin=sympy.Rational(3, 2))
            >>> hs.multiplicity == 4 == hs.dimension
            True
        """
        return int(2 * self._spin) + 1


def SpinBasisKet(*numer_denom, hs):
    """Constructor for a :class:`BasisKet` for a :class:`SpinSpace`

    For a half-integer spin system::

        >>> hs = SpinSpace('s', spin=(3, 2))
        >>> assert SpinBasisKet(1, 2, hs=hs) == BasisKet("+1/2", hs=hs)

    For an integer spin system::

        >>> hs = SpinSpace('s', spin=1)
        >>> assert SpinBasisKet(1, hs=hs) == BasisKet("+1", hs=hs)

    Note that ``BasisKet(1, hs=hs)`` with an integer index (which would
    hypothetically refer to ``BasisKet("0", hs=hs)`` is not allowed for spin
    systems::

        >>> BasisKet(1, hs=hs)
        Traceback (most recent call last):
            ...
        TypeError: label_or_index must be an instance of one of str, SpinIndex; not int

    Raises:
        TypeError: if `hs` is not a :class:`SpinSpace` or the wrong number of
            positional arguments is given
        ValueError: if any of the positional arguments are out range for the
            given `hs`
    """
    try:
        spin_numer, spin_denom = hs.spin.as_numer_denom()
    except AttributeError:
        raise TypeError(
            "hs=%s for SpinBasisKet must be a SpinSpace instance" % hs)
    assert spin_denom in (1, 2)
    if spin_denom == 1:  # integer spin
        if len(numer_denom) != 1:
            raise TypeError(
                "SpinBasisKet requires exactly one positional argument for an "
                "integer-spin Hilbert space")
        numer = numer_denom[0]
        if numer < -spin_numer or numer > spin_numer:
            raise ValueError(
                "spin quantum number %s must be in range (%s, %s)"
                % (numer, -spin_numer, spin_numer))
        label = str(numer)
        if numer > 0:
            label = "+" + label
        return BasisKet(label, hs=hs)
    else:  # half-integer spin
        if len(numer_denom) != 2:
            raise TypeError(
                "SpinBasisKet requires exactly two positional arguments for a "
                "half-integer-spin Hilbert space")
        numer, denom = numer_denom
        numer = int(numer)
        denom = int(denom)
        if denom != 2:
            raise ValueError(
                "The second positional argument (denominator of the spin "
                "quantum number) must be 2, not %s" % denom)
        if numer < -spin_numer or numer > spin_numer:
            raise ValueError(
                "spin quantum number %s/%s must be in range (%s/2, %s/2)"
                % (numer, denom, -spin_numer, spin_numer))
        label = str(numer)
        if numer > 0:
            label = "+" + label
        label = label + "/2"
        return BasisKet(label, hs=hs)
