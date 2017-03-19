#This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QNET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QNET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################

import pytest

from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, ProductSpace, TrivialSpace, FullSpace)
from qnet.algebra.operator_algebra import Destroy
from qnet.algebra.state_algebra import KetSymbol, BasisKet, TensorKet


def test_instantiate_with_basis():
    """Test that a local space can be instantiated with an explicit basis"""
    hs1 = LocalSpace('1', basis=(0, 1))
    assert hs1.dimension == 2
    assert hs1.basis == ('0', '1')
    hs1 = LocalSpace('1', basis=['g', 'e'])
    assert hs1.dimension == 2
    assert hs1.basis == ('g', 'e')


def test_basis_change():
    """Test that we can change the basis of an Expression's Hilbert space
    through substitution"""
    a = Destroy(hs=1)
    assert a.space == LocalSpace('1')
    assert a.space.basis is None
    assert a.space.dimension is None
    subs = {LocalSpace('1'): LocalSpace('1', basis=(-1, 0, 1))}
    b = a.substitute(subs)
    assert str(a) == str(b)
    assert a != b
    assert b.space.dimension == 3
    assert b.space.basis == ('-1', '0', '1')


def test_op_product_space():
    """Test that a product of operators has the correct Hilbert space"""
    a = Destroy(hs=1)
    b = Destroy(hs=2)
    p = a * b
    assert p.space == ProductSpace(LocalSpace(1), LocalSpace(2))
    assert p.space.dimension is None
    assert p.space.basis is None

    hs1 = LocalSpace(1, dimension=3)
    a = a.substitute({LocalSpace(1): hs1})
    p = a * b
    assert p.space == ProductSpace(hs1, LocalSpace(2))
    assert p.space.dimension is None
    assert p.space.basis is None

    hs2 = LocalSpace(2, dimension=2)
    b = b.substitute({LocalSpace(2): hs2})
    p = a * b
    ps = ProductSpace(hs1, hs2)
    assert p.space == ps
    assert p.space.dimension == 6
    assert p.space.basis == ('0,0', '0,1', '1,0', '1,1', '2,0', '2,1')

    hs1_2 = LocalSpace(1, basis=('g', 'e'))
    hs2_2 = LocalSpace(2, basis=('g', 'e'))
    p = p.substitute({hs1: hs1_2, hs2: hs2_2})
    assert p.space.dimension == 4
    assert p.space.basis == ('g,g', 'g,e', 'e,g', 'e,e')

    b = b.substitute({hs2: hs1})
    p = a * b
    assert p.space == hs1
    assert p.space.dimension == 3
    assert p.space.basis == ('0', '1', '2')


def test_ket_product_space():
    """Test that the product of two kets has the correct Hilbert space"""
    a = KetSymbol('0', hs=1)
    b = KetSymbol('0', hs=2)
    p = a * b
    assert p.space == ProductSpace(LocalSpace(1), LocalSpace(2))
    assert p.space.dimension is None
    assert p.space.basis is None

    hs1 = LocalSpace(1, dimension=3)
    a = a.substitute({LocalSpace(1): hs1})
    p = a * b
    assert p.space == ProductSpace(hs1, LocalSpace(2))
    assert p.space.dimension is None
    assert p.space.basis is None

    hs2 = LocalSpace(2, dimension=2)
    b = b.substitute({LocalSpace(2): hs2})
    p = a * b
    ps = ProductSpace(hs1, hs2)
    assert p.space == ps
    assert p.space.dimension == 6
    assert p.space.basis == ('0,0', '0,1', '1,0', '1,1', '2,0', '2,1')


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
    assert h3.dimension is None
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
    hs1 = LocalSpace('1', basis=['g', 'e'])
    hs2 = LocalSpace('2', dimension=2)
    hs3 = LocalSpace('3', dimension=2)
    hs4 = LocalSpace('4', dimension=2)

    g_1, e_1 = hs1.basis_states
    assert g_1 == BasisKet('g', hs=hs1)
    assert e_1 == BasisKet('e', hs=hs1)

    zero_2, one_2 = hs2.basis_states
    assert zero_2 == BasisKet(0, hs=hs2)
    assert one_2 == BasisKet(1, hs=hs2)

    hs_prod = hs1 * hs2
    g0, g1, e0, e1 = list(hs_prod.basis_states)
    assert g0 == g_1 * zero_2
    assert g1 == g_1 * one_2
    assert e0 == e_1 * zero_2
    assert e1 == e_1 * one_2

    hs_prod4 = hs1 * hs2 * hs3 * hs4
    basis = hs_prod4.basis_states
    assert next(basis) == (BasisKet('g', hs=hs1) * BasisKet(0, hs=hs2) *
                           BasisKet(0, hs=hs3) * BasisKet(0, hs=hs4))
    assert next(basis) == (BasisKet('g', hs=hs1) * BasisKet(0, hs=hs2) *
                           BasisKet(0, hs=hs3) * BasisKet(1, hs=hs4))
    assert next(basis) == (BasisKet('g', hs=hs1) * BasisKet(0, hs=hs2) *
                           BasisKet(1, hs=hs3) * BasisKet(0, hs=hs4))
