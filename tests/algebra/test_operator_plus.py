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
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import (
    OperatorSymbol, ZeroOperator, IdentityOperator, OperatorPlus)
from sympy import symbols
from sympy import S


def test_op_plus_scalar():
    """Test the we can add a scalar to an operator"""
    hs = LocalSpace("0")
    A = OperatorSymbol('A', hs=hs)
    alpha = symbols('alpha')
    assert A + 0 == A
    assert OperatorPlus.create(A, 0) == A
    assert 0 + A == A
    assert A + S.Zero == A
    assert ZeroOperator + S.Zero == ZeroOperator
    assert OperatorPlus.create(ZeroOperator, S.Zero) == ZeroOperator
    assert A + S.One == A + 1
    assert A + alpha == OperatorPlus(alpha * IdentityOperator, A)
    assert (OperatorPlus.create(alpha, A) ==
            OperatorPlus(alpha * IdentityOperator, A))
