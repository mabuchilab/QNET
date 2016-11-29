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

from qnet.circuit_components.displace_cc import Displace


def test_displace_uniqueness():
    """Test that different instantiations of a Displace component are poperly
    recognized as different"""
    W1 = Displace('W', alpha=(-79.838356622-35.806239846j))
    W2 = Displace('W', alpha=-79.838356622)
    W3 = Displace('W')
    assert 'alpha' in W1.kwargs
    assert hash(W1) != hash(W2)
    assert W1 != W2
    assert W2 != W3
    assert W1 != W3
