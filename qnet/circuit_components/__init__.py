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

"""
This module contains all defined *primitive* circuit component definitions as well as the compiled circuit definitions
that are automatically created via the ``$QNET/bin/parse_qhdl.py`` script.
For some examples on how to create your own circuit definition file, check out the source code to

    * :py:mod:`qnet.circuit_components.single_sided_jaynes_cummings_cc`
    * :py:mod:`qnet.circuit_components.three_port_opo_cc`
    * :py:mod:`qnet.circuit_components.kerr_cavity_cc`

The module :py:mod:`qnet.circuit_components.component` features some base classes for component definitions and the module :py:mod:`qnet.circuit_components.library`
features some utility functions to help manage the circuit component definitions.
"""
