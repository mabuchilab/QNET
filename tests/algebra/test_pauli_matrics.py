"""Test for PauliX, PauliY, PauliZ"""
from sympy import I

import pytest

from qnet import (
    PauliX, PauliY, PauliZ, LocalSigma, LocalSpace, LocalProjector, SpinSpace)


def test_fock_pauli_matrices():
    """Test correctness of Pauli matrices on a Fock space"""
    assert PauliX(1) == LocalSigma(0, 1, hs=1) + LocalSigma(1, 0, hs=1)
    assert PauliX(1) == PauliX('1') == PauliX(LocalSpace('1'))
    assert PauliY(1).expand() == (
        -I * LocalSigma(0, 1, hs=1) + I * LocalSigma(1, 0, hs=1))
    assert PauliY(1) == PauliY('1') == PauliY(LocalSpace('1'))
    assert PauliZ(1) == LocalProjector(0, hs=1) - LocalProjector(1, hs=1)
    assert PauliZ(1) == PauliZ('1') == PauliZ(LocalSpace('1'))
    assert PauliX(1, states=(0, 2)) == (
        LocalSigma(0, 2, hs=1) + LocalSigma(2, 0, hs=1))

    hs = LocalSpace("1", basis=('g', 'e', 'r'))
    assert PauliX(hs) == LocalSigma(0, 1, hs=hs) + LocalSigma(1, 0, hs=hs)
    assert PauliX(hs) == PauliX(hs, states=('g', 'e'))
    assert PauliY(hs).expand() == (
        -I * LocalSigma(0, 1, hs=hs) + I * LocalSigma(1, 0, hs=hs))
    assert PauliY(hs) == PauliY(hs, states=('g', 'e'))
    assert PauliZ(hs) == LocalProjector(0, hs=hs) - LocalProjector(1, hs=hs)
    assert PauliZ(hs) == PauliZ(hs, states=('g', 'e'))
    assert PauliX(hs, states=(0, 2)) == (
        LocalSigma('g', 'r', hs=hs) + LocalSigma('r', 'g', hs=hs))
    assert PauliX(hs, states=(0, 2)) == PauliX(hs, states=('g', 'r'))


def test_spin_pauli_matrices():
    """Test correctness of Pauli matrices on a spin space"""
    hs = SpinSpace("s", spin='1/2', basis=('down', 'up'))
    assert PauliX(hs) == (
        LocalSigma('down', 'up', hs=hs) + LocalSigma('up', 'down', hs=hs))
    assert PauliX(hs) == PauliX(hs, states=('down', 'up'))
    assert PauliY(hs).expand() == (
        -I * LocalSigma('down', 'up', hs=hs) +
        I * LocalSigma('up', 'down', hs=hs))
    assert PauliY(hs) == PauliY(hs, states=('down', 'up'))
    assert PauliZ(hs) == (
        LocalProjector('down', hs=hs) - LocalProjector('up', hs=hs))
    assert PauliZ(hs) == PauliZ(hs, states=('down', 'up'))

    hs = SpinSpace("s", spin=1, basis=('-', '0', '+'))
    with pytest.raises(TypeError):
        PauliX(hs, states=(0, 2))
    assert PauliX(hs, states=('-', '+')) == (
        LocalSigma('-', '+', hs=hs) + LocalSigma('+', '-', hs=hs))
    assert PauliX(hs) == PauliX(hs, states=('-', '0'))
