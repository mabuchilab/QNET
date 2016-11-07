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
Component definition file for an all-optical Relay model.
See documentation of :py:class:`Relay`.
"""
from qnet.circuit_components.component import Component, SubComponent
from qnet.algebra.circuit_algebra import SLH, Matrix
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import LocalProjector, LocalSigma


class RelayDoubleProbe(Component):
    """This is the Relay model as used in our group's QEC papers
    [#qec1]_,[#qec2]_.  The ``SET`` and ``RESET`` inputs control whether the
    ``POW`` input is routed through the ``OUT`` or  the ``NOUT`` output port.

    Since the scattering matrix is of block diagonal form (2x2,2x2) we provide
    sub component models for the individual subsystems :py:class:`RelayOut` and
    :py:class:`RelayControl`.

    .. [#qec1] http://pra.aps.org/abstract/PRA/v80/i4/e045802
    .. [#qec2] http://prl.aps.org/abstract/PRL/v105/i4/e040502
    """

    CDIM = 6

    PORTSIN = ['POW1', 'VIn1', 'POW2', 'VIn2', 'SET', 'RESET']
    PORTSOUT = ['NOUT1', 'OUT1','NOUT2', 'OUT2', 'UOut1', 'UOut2']

    sub_blockstructure = (2, 2, 2)


    @property
    def space(self):
        return LocalSpace(self.name, basis = ("h", "g"))

    def _creduce(self):
        return RelayOut(self, 0) + RelayOut(self, 1) + RelayControl(self, 2)

    def _toSLH(self):
        return self.creduce().toSLH()



class RelayOut(SubComponent):
    """
    First subcomponent of a :py:class:`Relay` model.
    """


    def _toSLH(self):

        Pi_g = LocalProjector(self.space, 'g')
        Pi_h = LocalProjector(self.space, 'h')

        S = Matrix([[Pi_g, -Pi_h ], [-Pi_h, Pi_g]])
        return SLH(S, Matrix([[0]]*2), 0)

class RelayControl(SubComponent):
    """
    Second subcomponent of a :py:class:`Relay` model.
    """


    def _toSLH(self):
        Pi_g = LocalProjector(self.space, 'g')
        Pi_h = LocalProjector(self.space, 'h')

        sigma_gh = LocalSigma(self.space, 'g', 'h')
        sigma_hg = LocalSigma(self.space, 'h', 'g')

        S = Matrix([[Pi_g, - sigma_hg ], [-sigma_gh, Pi_h]])
        return SLH(S, Matrix([[0]]*2), 0)
