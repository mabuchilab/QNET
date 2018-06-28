"""Constructors for Pauli-Matrix operators on any two levels of a system"""
from sympy import I

from ..core.hilbert_space_algebra import LocalSpace
from ..core.operator_algebra import LocalSigma, LocalProjector


__all__ = ['PauliX', 'PauliY', 'PauliZ']


def _get_pauli_args(local_space, states):
    if isinstance(local_space, (str, int)):
        local_space = LocalSpace(local_space)
    if states is None:
        if local_space.has_basis:
            states = local_space.basis_labels[0:2]
        else:
            states = (0, 1)
    return local_space, states


def PauliX(local_space, states=None):
    r"""Pauli-type X-operator

    .. math::

        \hat{\sigma}_x = \begin{pmatrix}
            0 & 1 \\
            1 & 0
        \end{pmatrix}

    on an arbitrary two-level system.

    Args:
        local_space (str or int or .LocalSpace): Associated Hilbert space.
            If :class:`str` or :class:`int`, a :class:`LocalSpace` with a
            matching label will be created.
        states (None or tuple[int or str]): The labels for the basis states
            for the two levels on which the operator acts. If None, the two
            lowest levels are used.

    Returns:
        Operator: Local X-operator as a linear combination of
        :class:`LocalSigma`
    """
    local_space, states = _get_pauli_args(local_space, states)
    g, e = states
    return (
        LocalSigma.create(g, e, hs=local_space) +
        LocalSigma.create(e, g, hs=local_space))


def PauliY(local_space, states=None):
    r""" Pauli-type Y-operator

    .. math::

        \hat{\sigma}_x = \begin{pmatrix}
            0 & -i \\
            i & 0
        \end{pmatrix}

    on an arbitrary two-level system.


    See :func:`PauliX`
    """
    local_space, states = _get_pauli_args(local_space, states)
    g, e = states
    return I * (-LocalSigma.create(g, e, hs=local_space) +
                LocalSigma.create(e, g, hs=local_space))


def PauliZ(local_space, states=None):
    r"""Pauli-type Z-operator

    .. math::

        \hat{\sigma}_x = \begin{pmatrix}
            1 & 0 \\
            0 & -1
        \end{pmatrix}

    on an arbitrary two-level system.

    See :func:`PauliX`
    """
    local_space, states = _get_pauli_args(local_space, states)
    g, e = states
    return (
        LocalProjector(g, hs=local_space) - LocalProjector(e, hs=local_space))
