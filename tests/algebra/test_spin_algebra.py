"""Test the spin algebra"""
import pytest

from qnet import SpinSpace, SpinBasisKet, LocalSpace


def test_spin_basis_ket():
    """Test the properties of BasisKet for the example of a spin system"""
    hs = SpinSpace('s', spin=(3, 2))
    ket_lowest = SpinBasisKet(-3, 2, hs=hs)
    assert ket_lowest.index == 0
    assert ket_lowest.next() == SpinBasisKet(-1, 2, hs=hs)
    assert ket_lowest.next().prev() == ket_lowest
    assert ket_lowest.next(n=2).prev(n=2) == ket_lowest
    assert ket_lowest.prev().is_zero
    assert SpinBasisKet(3, 2, hs=hs).next().is_zero
    assert SpinBasisKet(3, 2, hs=hs).index == 3


def test_tls_invalid_basis():
    """Test that trying to instantiate a TLS with canonical basis labes in the
    wrong order raises a ValueError"""
    with pytest.raises(ValueError) as exc_info:
        SpinSpace('tls', spin='1/2', basis=('up', 'down'))
    assert "Invalid basis" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinSpace('tls', spin='1/2', basis=('+', '-'))
    assert "Invalid basis" in str(exc_info.value)


def test_invalid_spin_basis_ket():
    """Test that trying to instantiate invalid :func:`SpinBasisKet` raises the
    appropriate exceptions"""
    hs1 = SpinSpace('s', spin='3/2')
    hs2 = SpinSpace('s', spin=1)
    with pytest.raises(TypeError) as exc_info:
        SpinBasisKet(1, 2, hs=LocalSpace(1))
    assert "must be a SpinSpace" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        SpinBasisKet(1, hs=hs1)
    assert "exactly two positional arguments" in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        SpinBasisKet(1, 2, hs=hs2)
    assert "exactly one positional argument" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinBasisKet(1, 3, hs=hs1)
    assert "must be 2" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinBasisKet(-5, 2, hs=hs1)
    assert "must be in range" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinBasisKet(5, 2, hs=hs1)
    assert "must be in range" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinBasisKet(-5, hs=hs2)
    assert "must be in range" in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        SpinBasisKet(5, hs=hs2)
    assert "must be in range" in str(exc_info.value)
