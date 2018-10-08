import pytest

from sympy import Rational, symbols

from qnet.algebra.core.hilbert_space_algebra import (
    LocalSpace, ProductSpace, TrivialSpace, FullSpace)
from qnet.algebra.core.exceptions import BasisNotSetError
from qnet.algebra.library.fock_operators import Destroy
from qnet.algebra.library.spin_algebra import SpinSpace, SpinBasisKet
from qnet.algebra.core.state_algebra import (
    KetSymbol, BasisKet, TrivialKet)
from qnet.utils.indices import IdxSym, StrLabel

def test_instantiate_with_basis():
    """Test that a local space can be instantiated with an explicit basis"""
    hs1 = LocalSpace('1', basis=(0, 1))
    assert hs1.dimension == 2
    assert hs1.basis_labels == ('0', '1')
    hs1 = LocalSpace('1', basis=['g', 'e'])
    assert hs1.dimension == 2
    assert hs1.basis_labels == ('g', 'e')


def test_basis_change():
    """Test that we can change the basis of an Expression's Hilbert space
    through substitution"""
    a = Destroy(hs=1)
    assert a.space == LocalSpace('1')
    assert not a.space.has_basis
    subs = {LocalSpace('1'): LocalSpace('1', basis=(-1, 0, 1))}
    b = a.substitute(subs)
    assert str(a) == str(b)
    assert a != b
    assert b.space.dimension == 3
    assert b.space.basis_labels == ('-1', '0', '1')


def test_op_product_space():
    """Test that a product of operators has the correct Hilbert space"""
    a = Destroy(hs=1)
    b = Destroy(hs=2)
    p = a * b
    assert p.space == ProductSpace(LocalSpace(1), LocalSpace(2))
    assert not p.space.has_basis

    hs1 = LocalSpace(1, dimension=3)
    a = a.substitute({LocalSpace(1): hs1})
    p = a * b
    assert p.space == ProductSpace(hs1, LocalSpace(2))
    assert not p.space.has_basis

    hs2 = LocalSpace(2, dimension=2)
    b = b.substitute({LocalSpace(2): hs2})
    p = a * b
    ps = ProductSpace(hs1, hs2)
    assert p.space == ps
    assert p.space.dimension == 6
    assert p.space.basis_labels == ('0,0', '0,1', '1,0', '1,1', '2,0', '2,1')

    hs1_2 = LocalSpace(1, basis=('g', 'e'))
    hs2_2 = LocalSpace(2, basis=('g', 'e'))
    p = p.substitute({hs1: hs1_2, hs2: hs2_2})
    assert p.space.dimension == 4
    assert p.space.basis_labels == ('g,g', 'g,e', 'e,g', 'e,e')

    b = b.substitute({hs2: hs1})
    p = a * b
    assert p.space == hs1
    assert p.space.dimension == 3
    assert p.space.basis_labels == ('0', '1', '2')


def test_ket_product_space():
    """Test that the product of two kets has the correct Hilbert space"""
    a = KetSymbol('0', hs=1)
    b = KetSymbol('0', hs=2)
    p = a * b
    assert p.space == ProductSpace(LocalSpace(1), LocalSpace(2))
    assert not p.space.has_basis

    hs1 = LocalSpace(1, dimension=3)
    a = a.substitute({LocalSpace(1): hs1})
    p = a * b
    assert p.space == ProductSpace(hs1, LocalSpace(2))
    assert not p.space.has_basis

    hs2 = LocalSpace(2, dimension=2)
    b = b.substitute({LocalSpace(2): hs2})
    p = a * b
    ps = ProductSpace(hs1, hs2)
    assert p.space == ps
    assert p.space.dimension == 6
    assert p.space.basis_labels == ('0,0', '0,1', '1,0', '1,1', '2,0', '2,1')


def test_product_space():

    # create HilbertSpaces
    h1 = LocalSpace("h1")
    h2 = LocalSpace("h2")
    h3 = LocalSpace("h3")

    # productspace
    assert h1 * h2 == ProductSpace(h1, h2)
    assert h3 * h1 * h2 == ProductSpace(h1, h2, h3)

    # space "subtraction/division/cancellation"
    assert (h1 * h2) / h1 == h2
    assert (h1 * h2 * h3) / h1 == h2 * h3
    assert (h1 * h2 * h3) / (h1 * h3) == h2

    # space "intersection"
    assert (h1 * h2) & h1 == h1
    assert (h1 * h2 * h3) & h1 == h1
    assert h1 * h1 == h1


def test_dimension():
    h1 = LocalSpace("h1", dimension = 10)
    h2 = LocalSpace("h2", dimension = 20)
    h3 = LocalSpace("h3")
    h4 = LocalSpace("h4", dimension = 100)

    assert (h1*h2).dimension == h1.dimension * h2.dimension
    with pytest.raises(BasisNotSetError):
        h3.dimension
    assert h4.dimension == 100


def test_space_ordering():
    h1 = LocalSpace("h1")
    h2 = LocalSpace("h2")
    h3 = LocalSpace("h3")

    assert h1 <= h1
    assert h1 <= (h1 * h2)
    assert not (h1 <= h2)
    assert not (h1 < h1)
    assert TrivialSpace < h1 < FullSpace
    assert h1>= h1
    assert h1 * h2 > h2
    assert not (h1 * h2 > h3)


def test_operations():
    h1 = LocalSpace("h1")
    h2 = LocalSpace("h2")
    h3 = LocalSpace("h3")

    h123 = h1 * h2 * h3
    h12 = h1 * h2
    h23 = h2 * h3
    h13 = h1 * h3
    assert h12 * h13 == h123
    assert h12 / h13 == h2
    assert h12 & h13 == h1
    assert (h12 / h13) * (h13 & h12) == h12
    assert h1 & h12 == h1


def test_hs_basis_states():
    """Test that we can obtain the basis states of a Hilbert space"""
    hs0 = LocalSpace('0')
    hs1 = LocalSpace('1', basis=['g', 'e'])
    hs2 = LocalSpace('2', dimension=2)
    hs3 = LocalSpace('3', dimension=2)
    hs4 = LocalSpace('4', dimension=2)
    spin = SpinSpace('s', spin=Rational(3/2))

    assert isinstance(hs0.basis_state(0), BasisKet)
    assert isinstance(hs0.basis_state(1), BasisKet)
    with pytest.raises(BasisNotSetError):
        _ = hs0.basis_state('0')
    with pytest.raises(BasisNotSetError):
        _ = hs0.dimension

    g_1, e_1 = hs1.basis_states
    assert hs1.dimension == 2
    assert hs1.basis_labels == ('g', 'e')
    assert g_1 == BasisKet('g', hs=hs1)
    assert e_1 == BasisKet('e', hs=hs1)
    assert g_1 == hs1.basis_state('g')
    assert g_1 == hs1.basis_state(0)
    assert e_1 == hs1.basis_state(1)
    with pytest.raises(IndexError):
        hs1.basis_state(2)
    with pytest.raises(KeyError):
        hs1.basis_state('r')

    zero_2, one_2 = hs2.basis_states
    assert zero_2 == BasisKet(0, hs=hs2)
    assert one_2 == BasisKet(1, hs=hs2)

    hs_prod = hs1 * hs2
    g0, g1, e0, e1 = list(hs_prod.basis_states)
    assert g0 == g_1 * zero_2
    assert g1 == g_1 * one_2
    assert e0 == e_1 * zero_2
    assert e1 == e_1 * one_2
    assert g0 == hs_prod.basis_state(0)
    assert e1 == hs_prod.basis_state(3)
    assert e1 == hs_prod.basis_state('e,1')
    assert hs_prod.basis_labels == ('g,0', 'g,1', 'e,0', 'e,1')
    with pytest.raises(IndexError):
        hs_prod.basis_state(4)
    with pytest.raises(KeyError):
        hs_prod.basis_state('g0')

    hs_prod4 = hs1 * hs2 * hs3 * hs4
    basis = hs_prod4.basis_states
    assert next(basis) == (BasisKet('g', hs=hs1) * BasisKet(0, hs=hs2) *
                           BasisKet(0, hs=hs3) * BasisKet(0, hs=hs4))
    assert next(basis) == (BasisKet('g', hs=hs1) * BasisKet(0, hs=hs2) *
                           BasisKet(0, hs=hs3) * BasisKet(1, hs=hs4))
    assert next(basis) == (BasisKet('g', hs=hs1) * BasisKet(0, hs=hs2) *
                           BasisKet(1, hs=hs3) * BasisKet(0, hs=hs4))

    assert TrivialSpace.dimension == 1
    assert list(TrivialSpace.basis_states) == [TrivialKet, ]
    assert TrivialSpace.basis_state(0) == TrivialKet

    basis = spin.basis_states
    ket = next(basis)
    assert ket == BasisKet('-3/2', hs=spin)
    with pytest.raises(TypeError):
        ket == BasisKet(0, hs=spin)
    assert next(basis) == SpinBasisKet(-1, 2, hs=spin)
    assert next(basis) == SpinBasisKet(+1, 2, hs=spin)
    assert next(basis) == SpinBasisKet(+3, 2, hs=spin)

    with pytest.raises(BasisNotSetError):
        FullSpace.dimension
    with pytest.raises(BasisNotSetError):
        FullSpace.basis_states


def test_hilbertspace_free_symbols():
    """Test that Hilbert spaces with an indexed name return the index symbol in
    free_symbols"""
    i, j = symbols('i, j', cls=IdxSym)
    assert LocalSpace(1).free_symbols == set()
    hs_i = LocalSpace(StrLabel(i))
    hs_j = LocalSpace(StrLabel(j))
    assert hs_i.free_symbols == {i}
    assert hs_j.free_symbols == {j}
    assert (hs_i * hs_j).free_symbols == {i, j}
