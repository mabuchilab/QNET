"""Test qnet.algebra.library.hilbert_spaces"""
from qnet import SpinSpace, IdxSym

import sympy

import pytest


def test_invalid_spin_space():
    """Test that a ValueError is raised when trying to instantiate a SpinSpace
    with and invalid `spin` value"""
    with pytest.raises(ValueError) as exc_info:
        SpinSpace(0, spin=0)
    assert "must be greater than zero" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinSpace(0, spin=-1)
    assert "must be greater than zero" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinSpace(0, spin=sympy.Rational(2, 3))
    assert "must be an integer or half-integer" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinSpace(0, spin="one")
    assert "must be an integer or half-integer" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinSpace(0, spin=IdxSym('i'))
    assert "must be an integer or half-integer" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        SpinSpace(0)
    assert "required keyword-only argument: 'spin'" in str(exc_info.value)
