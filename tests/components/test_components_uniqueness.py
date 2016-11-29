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
from qnet.circuit_components.two_port_kerr_cavity_cc import TwoPortKerrCavity


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


def test_subcomponents_uniqueness():
    """Test that two "identical" subcomponent from two different components do
    not compare as identical"""
    C1 = TwoPortKerrCavity('Kerr', Delta=1.0)
    C2 = TwoPortKerrCavity('Kerr', Delta=2.0)
    assert C1 != C2
    assert C1.port1 != C2.port1
    assert C1.port2 != C2.port2
