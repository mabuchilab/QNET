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
"""
This module contains all defined *primitive* circuit component definitions as
well as the compiled circuit definitions
that are automatically created via the ``$QNET/bin/parse_qhdl.py`` script.
For some examples on how to create your own circuit definition file, check out
the source code to

    * :mod:`qnet.circuit_components.single_sided_jaynes_cummings_cc`
    * :mod:`qnet.circuit_components.three_port_opo_cc`
    * :mod:`qnet.circuit_components.kerr_cavity_cc`

The module :py:mod:`qnet.circuit_components.component` features some base
classes for component definitions and the module
:py:mod:`qnet.circuit_components.library`
features some utility functions to help manage the circuit component
definitions.

.. note::

    For convenience in an interactive context, this package is also exposed as
    :mod:`qnet.cc <qnet.circuit_components>`. This works only if :mod:`qnet`
    was previously imported.
"""

import qnet.circuit_components.component
import qnet.circuit_components.library

from .component import *
# qnet.circuit_components.library is not exposed in a flat API
from .and_cc import *
from .beamsplitter_cc import *
from .delay_cc import *
from .displace_cc import *
from .double_sided_jaynes_cummings_cc import *
from .double_sided_opo_cc import *
from .inverting_fanout_cc import *
from .kerr_cavity_cc import *
from .latch_cc import *
from .linear_cavity_cc import *
from .mach_zehnder_cc import *
from .open_lossy_cc import *
from .phase_cc import *
from .pseudo_nand_cc import *
from .pseudo_nand_latch_cc import *
from .relay_cc import *
from .relay_double_probe_cc import *
from .single_sided_jaynes_cummings_cc import *
from .single_sided_opo_cc import *
from .three_port_kerr_cavity_cc import *
from .three_port_opo_cc import *
from .two_port_kerr_cavity_cc import *
from .z_probe_cavity_cc import *

from qnet._flat_api_tools import _combine_all
__all__ = _combine_all(
    'qnet.circuit_components.component',
    'qnet.circuit_components.and_cc',
    'qnet.circuit_components.beamsplitter_cc',
    'qnet.circuit_components.delay_cc',
    'qnet.circuit_components.displace_cc',
    'qnet.circuit_components.double_sided_jaynes_cummings_cc',
    'qnet.circuit_components.double_sided_opo_cc',
    'qnet.circuit_components.inverting_fanout_cc',
    'qnet.circuit_components.kerr_cavity_cc',
    'qnet.circuit_components.latch_cc',
    'qnet.circuit_components.linear_cavity_cc',
    'qnet.circuit_components.mach_zehnder_cc',
    'qnet.circuit_components.open_lossy_cc',
    'qnet.circuit_components.phase_cc',
    'qnet.circuit_components.pseudo_nand_cc',
    'qnet.circuit_components.pseudo_nand_latch_cc',
    'qnet.circuit_components.relay_cc',
    'qnet.circuit_components.relay_double_probe_cc',
    'qnet.circuit_components.single_sided_jaynes_cummings_cc',
    'qnet.circuit_components.single_sided_opo_cc',
    'qnet.circuit_components.three_port_kerr_cavity_cc',
    'qnet.circuit_components.three_port_opo_cc',
    'qnet.circuit_components.two_port_kerr_cavity_cc',
    'qnet.circuit_components.z_probe_cavity_cc',
)
