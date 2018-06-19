"""Collection of operators that act on a :class:`.SpinSpace`"""
from abc import ABCMeta

from sympy import sympify, sqrt

from ..core.operator_algebra import LocalOperator
from .hilbert_spaces import SpinSpace


__all__ = ['SpinOperator', 'Jz', 'Jplus', 'Jminus']
__private__ = ['Jpjmcoeff', 'Jzjmcoeff', 'Jmjmcoeff']


class SpinOperator(LocalOperator, metaclass=ABCMeta):
    """Base class for Operators in a spin space"""
    _hs_cls = SpinSpace

    def __init__(self, *args, hs):
        super().__init__(*args, hs=hs)
        if not isinstance(self.space, SpinSpace):
            raise TypeError(
                "hs %s must be an instance of SpinSpace" % self.space)


class Jz(SpinOperator):
    """$\Op{J}_z$ is the $z$ component of a general spin operator acting
    on a particular :class:`.LocalSpace` `hs` of freedom with well defined spin
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


class Jplus(SpinOperator):
    """ $\Op{J}_{+} = \Op{J}_x + i \op{J}_y$ is the raising ladder operator
    of a general spin operator acting on a particular :class:`.LocalSpace` `hs`
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


class Jminus(SpinOperator):
    """$\Op{J}_{-} = \Op{J}_x - i \op{J}_y$ is lowering ladder operator of a
    general spin operator acting on a particular :class:`.LocalSpace` `hs`
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


def Jpjmcoeff(ls, m, shift=False):
    r'''Eigenvalue of the $\Op{J}_{+}$ (:class:`Jplus`) operator, as a Sympy
    expression.

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


def Jzjmcoeff(ls, m, shift):
    r'''Eigenvalue of the $\Op{J}_z$ (:class:`Jz`) operator, as a Sympy
    expression.

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


def Jmjmcoeff(ls, m, shift):
    r'''Eigenvalue of the $\Op{J}_{-}$ (:class:`Jminus`) operator, as a Sympy
    expression

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
