# coding=utf-8
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
QNET contains symbolic algebra packages to handle:
 - quantum operators --> `qnet.algebra.operator_algebra`
 - quantum feedback circuits --> `qnet.algebra.circuit_algebra`
 - quantum super operators --> `qnet.algebra.super_operator_algebra`
 - quantum states --> `qnet.algebra.state_algebra`

It also features a parser for quantum feedback circuits specified
via Quantum Hardware Description Language:

    parse_qhdl.py -f MyCircuit.qhdl [-l|-L]

See our full docs for more information.
"""
__version__ = "1.2.1"