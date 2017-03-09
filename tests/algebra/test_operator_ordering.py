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
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################

import pytest

from qnet.algebra.operator_algebra import OperatorSymbol
from qnet.algebra.hilbert_space_algebra import LocalSpace


def test_operator_times_order():
    A1 = OperatorSymbol("A", hs=1)
    B1 = OperatorSymbol("B", hs=1)
    A2 = OperatorSymbol("A", hs=2)
    A3 = OperatorSymbol("A", hs=3)
    B4 = OperatorSymbol("B", hs=4)
    B1_m = OperatorSymbol("B", hs=LocalSpace(1, order_index=2))
    B2_m = OperatorSymbol("B", hs=LocalSpace(2, order_index=1))

    assert A1 * A2 == A2 * A1
    assert A1 * B1 != B1 * A1
    assert (A2 * A1).operands == (A1, A2)
    assert (B2_m * B1_m).operands == (B2_m, B1_m)
    assert ((B4+A3) * (A2+A1)).operands == (A1+A2, A3+B4)

