"""Definitions for an algebra on spin (angular momentum) Hilbert spaces, both
for integer and half-integer spin"""
from abc import ABCMeta
from collections.__init__ import OrderedDict

import sympy
from sympy import sympify, sqrt

from ..core.hilbert_space_algebra import LocalSpace
from ..core.state_algebra import BasisKet
from ..core.operator_algebra import LocalOperator, PseudoInverse
from ...utils.indices import SpinIndex

__all__ = [
    'SpinSpace', 'SpinBasisKet', 'SpinOperator', 'Jz', 'Jplus', 'Jminus']
__private__ = ['Jpjmcoeff', 'Jzjmcoeff', 'Jmjmcoeff']


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

    For convenience, you may also give `spin` as a tuple or a string::

        >>> hs = SpinSpace(label=0, spin=(3, 2))
        >>> assert hs == SpinSpace(label=0, spin=sympy.Rational(3, 2))
        >>> hs = SpinSpace(label=0, spin='3/2')
        >>> assert hs == SpinSpace(label=0, spin=(3, 2))

    You may use custom labels, e.g.::

        >>> hs = SpinSpace(label='s', spin='1/2', basis=('-', '+'))
        >>> hs.basis_labels
        ('-', '+')

    The labels "up" and "down" are recognized and printed as the appropritate
    arrow symbols::

        >>> hs = SpinSpace(label='s', spin='1/2', basis=('down', 'up'))
        >>> unicode(BasisKet('up', hs=hs))
        '|↑⟩⁽ˢ⁾'
        >>> unicode(BasisKet('down', hs=hs))
        '|↓⟩⁽ˢ⁾'

    Raises:

        ValueError: if `spin` is not an integer or half-integer greater than
            zero
    """
    _basis_label_types = (str, SpinIndex)  # acceptable types for labels

    def __init__(
            self, label, *, spin, basis=None, local_identifiers=None,
            order_index=None):
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
        if basis is None:
            basis = tuple([
                SpinIndex._static_render(bottom + n)
                for n in range(dimension)])
        else:
            # sometimes people don't think and use some of the "canonical" TLS
            # labels in the wrong order. We can catch it, so why not?
            if basis == ('up', 'down') or basis == ('+', '-'):
                raise ValueError(
                    "Invalid basis: you've switched %s and %s" % basis)
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


class SpinOperator(LocalOperator, metaclass=ABCMeta):
    """Base class for operators in a spin space"""
    _hs_cls = SpinSpace

    def __init__(self, *args, hs):
        super().__init__(*args, hs=hs)
        if not isinstance(self.space, SpinSpace):
            raise TypeError(
                "hs %s must be an instance of SpinSpace" % self.space)


class Jz(SpinOperator):
    """Spin (angular momentum) operator in z-direction

    $\Op{J}_z$ is the $z$ component of a general spin operator acting
    on a particular :class:`SpinSpace` `hs` of freedom with well defined spin
    quantum number $s$.  It is Hermitian::

        >>> hs = SpinSpace(1, spin=(1, 2))
        >>> print(ascii(Jz(hs=hs).adjoint()))
        J_z^(1)

    :class:`Jz`, :class:`Jplus` and :class:`Jminus` satisfy the angular
    momentum commutator algebra::

        >>> print(ascii((Jz(hs=hs) * Jplus(hs=hs) -
        ...              Jplus(hs=hs)*Jz(hs=hs)).expand()))
        J_+^(1)

        >>> print(ascii((Jz(hs=hs) * Jminus(hs=hs) -
        ...              Jminus(hs=hs)*Jz(hs=hs)).expand()))
        -J_-^(1)

        >>> print(ascii((Jplus(hs=hs) * Jminus(hs=hs)
        ...              - Jminus(hs=hs)*Jplus(hs=hs)).expand()))
        2 * J_z^(1)
        >>> Jplus(hs=hs).dag() == Jminus(hs=hs)
        True
        >>> Jminus(hs=hs).dag() == Jplus(hs=hs)
        True

    Printers should represent this operator with the default identifier::

        >>> Jz._identifier
        'J_z'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """

    _identifier = 'J_z'

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    def _adjoint(self):
        return self

    def _pseudo_inverse(self):
        return PseudoInverse(self)


class Jplus(SpinOperator):
    """Raising operator of a spin space

    $\Op{J}_{+} = \Op{J}_x + i \op{J}_y$ is the raising ladder operator
    of a general spin operator acting on a particular :class:`SpinSpace` `hs`
    with well defined spin quantum number $s$.  It's adjoint is the
    lowering operator::

        >>> hs = SpinSpace(1, spin=(1, 2))
        >>> print(ascii(Jplus(hs=hs).adjoint()))
        J_-^(1)

    :class:`Jz`, :class:`Jplus` and :class:`Jminus` satisfy that angular
    momentum commutator algebra, see :class:`Jz`

    Printers should represent this operator with the default identifier::

        >>> Jplus._identifier
        'J_+'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """

    _identifier = 'J_+'

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    def _adjoint(self):
        return Jminus(hs=self.space)

    def _pseudo_inverse(self):
        return PseudoInverse(self)


class Jminus(SpinOperator):
    """Lowering operator on a spin space

    $\Op{J}_{-} = \Op{J}_x - i \op{J}_y$ is the lowering ladder operator of
    a general spin operator acting on a particular :class:`SpinSpace` `hs`
    with well defined spin quantum number $s$.  It's adjoint is the raising
    operator::

        >>> hs = SpinSpace(1, spin=(1, 2))
        >>> print(ascii(Jminus(hs=hs).adjoint()))
        J_+^(1)

    :class:`Jz`, :class:`Jplus` and :class:`Jminus` satisfy that angular
    momentum commutator algebra, see :class:`Jz`.

    Printers should represent this operator with the default identifier::

        >>> Jminus._identifier
        'J_-'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """

    _identifier = 'J_-'

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    def _adjoint(self):
        return Jplus(hs=self.space)

    def _pseudo_inverse(self):
        return PseudoInverse(self)


def Jpjmcoeff(ls, m, shift=False) -> sympy.Expr:
    r'''Eigenvalue of the $\Op{J}_{+}$ (:class:`Jplus`) operator

    .. math::

        \Op{J}_{+} \ket{s, m} = \sqrt{s (s+1) - m (m+1)} \ket{s, m}

    where the multiplicity $s$ is implied by the size of the Hilbert space
    `ls`: there are $2s+1$ eigenstates with $m = -s, -s+1, \dots, s$.

    Args:
        ls (LocalSpace): The Hilbert space in which the $\Op{J}_{+}$ operator
            acts.
        m (str or int): If str, the label of the basis state of `hs` to which
            the operator is applied. If integer together with ``shift=True``,
            the zero-based index of the basis state. Otherwise, directly the
            quantum number $m$.
        shift (bool): If True for a integer value of `m`, treat `m` as the
            zero-based index of the basis state (i.e., shift `m` down by $s$ to
            obtain the quantum number $m$)
    '''
    assert isinstance(ls, SpinSpace)
    n = ls.dimension
    s = sympify(n - 1) / 2
    assert n == int(2 * s + 1)
    if isinstance(m, str):
        m = ls.basis_labels.index(m) - s  # m is now Sympy expression
    elif isinstance(m, int):
        if shift:
            assert 0 <= m < n
            m = m - s
    return sqrt(s * (s + 1) - m * (m + 1))


def Jzjmcoeff(ls, m, shift) -> sympy.Expr:
    r'''Eigenvalue of the $\Op{J}_z$ (:class:`Jz`) operator

    .. math::

        \Op{J}_{z} \ket{s, m} = m \ket{s, m}

    See also :func:`Jpjmcoeff`.
    '''
    assert isinstance(ls, SpinSpace)
    n = ls.dimension
    s = sympify(n - 1) / 2
    assert n == int(2 * s + 1)
    if isinstance(m, str):
        return ls.basis.index(m) - s
    elif isinstance(m, int):
        if shift:
            assert 0 <= m < n
            return m - s
    else:
        return sympify(m)


def Jmjmcoeff(ls, m, shift) -> sympy.Expr:
    r'''Eigenvalue of the $\Op{J}_{-}$ (:class:`Jminus`) operator

    .. math::

        \Op{J}_{-} \ket{s, m} = \sqrt{s (s+1) - m (m-1)} \ket{s, m}

    See also :func:`Jpjmcoeff`.
    '''
    assert isinstance(ls, SpinSpace)
    n = ls.dimension
    s = sympify(n - 1) / 2
    assert n == int(2 * s + 1)
    if isinstance(m, str):
        m = ls.basis.index(m) - s  # m is now Sympy expression
    elif isinstance(m, int):
        if shift:
            assert 0 <= m < n
            m = m - s
    return sqrt(s * (s + 1) - m * (m - 1))
