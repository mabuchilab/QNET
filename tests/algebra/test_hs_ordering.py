# This file is part of QNET.
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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################

import pytest

from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, TrivialSpace, FullSpace)

def test_product_space_order():
    H1 = LocalSpace(1)
    H2 = LocalSpace('2')
    assert H1 * H2 == H2 * H1
    assert (H1 * H2).operands == (H1, H2)

    H1 = LocalSpace(1)
    H2 = LocalSpace('2', order_index=2)
    H3 = LocalSpace(3, order_index=1)
    assert (H1 * H2 * H3).operands == (H3, H2, H1)
