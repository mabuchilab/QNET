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
Component definition file for a coherent field displacement component.
"""

from sympy.core.symbol import symbols

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import SLH
from qnet.algebra.matrix_algebra import Matrix


__all__ = ['Displace']


class Displace(Component):
    r"""Coherent displacement of the input field (usually vacuum) by a complex
    amplitude :math:`\alpha`.  This component serves as the model of an ideal
    laser source without internal non-classical internal dynamics.
    """

    CDIM = 1

    alpha = symbols('alpha')  # complex valued laser amplitude
    _parameters = ['alpha']

    PORTSIN = ["VacIn"]
    PORTSOUT = ["Out1"]

    def _toSLH(self):

        S = Matrix([[1]])
        L = Matrix([[self.alpha]])
        H = 0

        return SLH(S, L, H)
