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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################

import pytest

from qnet.algebra.circuit_algebra import SLH
from qnet.circuit_components.and_cc import And
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.delay_cc import Delay
from qnet.circuit_components.displace_cc import Displace
from qnet.circuit_components.double_sided_jaynes_cummings_cc import (
        DoubleSidedJaynesCummings)
from qnet.circuit_components.double_sided_opo_cc import DoubleSidedOPO
from qnet.circuit_components.inverting_fanout_cc import InvertingFanout
from qnet.circuit_components.kerr_cavity_cc import KerrCavity
from qnet.circuit_components.latch_cc import Latch
from qnet.circuit_components.linear_cavity_cc import LinearCavity
from qnet.circuit_components.mach_zehnder_cc import MachZehnder
from qnet.circuit_components.open_lossy_cc import OpenLossy
from qnet.circuit_components.phase_cc import Phase
from qnet.circuit_components.pseudo_nand_cc import PseudoNAND
from qnet.circuit_components.pseudo_nand_latch_cc import PseudoNANDLatch
from qnet.circuit_components.relay_cc import Relay
from qnet.circuit_components.relay_double_probe_cc import RelayDoubleProbe
from qnet.circuit_components.single_sided_jaynes_cummings_cc import (
        SingleSidedJaynesCummings, )
from qnet.circuit_components.single_sided_opo_cc import SingleSidedOPO
from qnet.circuit_components.three_port_kerr_cavity_cc import (
        ThreePortKerrCavity, )
from qnet.circuit_components.three_port_opo_cc import ThreePortOPO
from qnet.circuit_components.two_port_kerr_cavity_cc import TwoPortKerrCavity
from qnet.circuit_components.z_probe_cavity_cc import ZProbeCavity


components = [And, Beamsplitter, Delay, Displace, DoubleSidedJaynesCummings,
              DoubleSidedOPO, InvertingFanout, KerrCavity, LinearCavity,
              MachZehnder, OpenLossy, Phase, PseudoNAND, PseudoNANDLatch,
              Relay, RelayDoubleProbe, SingleSidedJaynesCummings,
              SingleSidedOPO, ThreePortKerrCavity, ThreePortOPO,
              TwoPortKerrCavity, ZProbeCavity]
# TODO: add Latch (slow)

@pytest.mark.parametrize('cls', components)
def test_component_interface(cls):
    """Test the basic interface that all components must fulfill"""
    comp = cls(name=str(cls.__name__))
    assert isinstance(comp, cls)
    comp = comp.creduce()

    if len(cls._parameters):
        pname = cls._parameters[0]
        name = str(cls.__name__)+'2'
        comp = cls(name=name, **{pname: 5})
        assert getattr(comp, pname) == 5
        assert comp.name == name
        for pname in cls._parameters:
            assert pname in comp.kwargs

    slh = comp.toSLH()
    assert isinstance(slh, SLH)
