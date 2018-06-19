"""Symbolic quantum and photonic circuit (SLH) algebra"""
import importlib

__all__ = ['init_algebra']


def init_algebra(*, default_hs_cls='LocalSpace'):
    """Initialize the algebra system

    Args:
        default_hs_cls (str): The name of the :class:`.LocalSpace` subclass
            that should be used when implicitly creating Hilbert spaces, e.g.
            in :class:`.OperatorSymbol`

    """
    from qnet.algebra.core.hilbert_space_algebra import LocalSpace
    from qnet.algebra.core.abstract_quantum_algebra import QuantumExpression
    default_hs_cls = getattr(importlib.import_module('qnet'), default_hs_cls)
    if issubclass(default_hs_cls, LocalSpace):
        QuantumExpression._default_hs_cls = default_hs_cls
    else:
        raise TypeError("default_hs_cls must be a subclass of LocalSpace")
    # TODO: init_algebra should eventually control which rules QNET uses.
