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
"""Component definition file for a infinite bandwidth beamsplitter with
variable mixing angle.  See :py:class:`Beamsplitter`
"""

from sympy import pi, sin, cos

from qnet.algebra.circuit_algebra import SLH
from qnet.algebra.operator_algebra import Matrix
from qnet.circuit_components.component import Component


class Beamsplitter(Component):
    r"""Infinite bandwidth beamsplitter model. It is a pure scattering
    component, i.e. it's internal dynamics are not modeled explicitly.  The
    single real parameter theta is the mixing angle for the two signals.  Note
    that there is an asymmetry in the effect on the two input signals due to
    the minus sign appearing in the scattering matrix

    .. math::

        S = \begin{pmatrix} \cos{\theta} & -\sin{\theta} \\ \sin{\theta} & \cos{\theta} \end{pmatrix}

    To achieve a more general beamsplitter combine this component with one or more
    :py:class:`qnet.circuit_components.Phase` components.

    Instantiate as::

        Beamsplitter("B", theta = pi/4)


    """
    CDIM = 2

    theta = pi/4 # mixing angle, default 50% mixing.
    _parameters = ['theta']

    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']

    def _toSLH(self):
        S = Matrix([[cos(self.theta), -sin(self.theta)],
                    [sin(self.theta),  cos(self.theta)]])
        L = Matrix([[0],
                    [0]])
        return SLH(S, L, 0)
