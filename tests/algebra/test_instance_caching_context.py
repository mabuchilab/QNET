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

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import OperatorSymbol, OperatorPlus
from qnet.algebra.abstract_algebra import (
    no_instance_caching, temporary_instance_cache)


def test_context_instance_caching():
    """Test that we can temporarily suppress instance caching"""
    h1 = LocalSpace("caching")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("a", hs=h1)
    c = OperatorSymbol("c", hs=h1)
    expr1 = a + b
    assert expr1 in OperatorPlus._instances.values()
    with no_instance_caching():
        assert expr1 in OperatorPlus._instances.values()
        expr2 = a + c
        assert expr2 not in OperatorPlus._instances.values()
    with temporary_instance_cache(OperatorPlus):
        assert len(OperatorPlus._instances) == 0
        expr2 = a + c
        assert expr2 in OperatorPlus._instances.values()
    assert expr1 in OperatorPlus._instances.values()
    assert expr2 not in OperatorPlus._instances.values()

