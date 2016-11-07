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
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, ProductSpace, TrivialSpace, FullSpace)
import pytest


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

