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
Component definition file for the Z-probe cavity model
from the Mabuchi-Lab Coherent Feedback Quantum Error Correction papers.

See documentation of :py:class:`ZProbeCavity`.
"""
from sympy.core.symbol import symbols
from sympy import sqrt

from qnet.algebra.circuit_algebra import SLH, Matrix
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import (
        identity_matrix, LocalProjector, Z, LocalSigma)
from qnet.circuit_components.component import Component, SubComponent


class ZProbeCavity(Component):
    """This is the Z-probe cavity model as used in our group's QEC papers
    [#qec1,#qec2]_ , which has three (dressed) internal states: r, g, h. The
    qubit is encoded in g,h, while r is used to drive transitions.  The first
    channel is the probe-signal, while the second and third channels are the
    two independent feedback beams.

    Since the scattering matrix is diagonal we provide sub component models for
    the individual subsystems: One :py:class:`ProbePort` and two
    :py:class:`FeedbackPort` instances..
    """
    CDIM = 5

    gamma = symbols('gamma', positive=True)   # decay rate into transverse modes
    gamma_p = symbols('gamma_p', positive=True)  # decay rate into transverse modes
    Delta = symbols('Delta', real=True)   # detuning between the cavity (mode) and the atomic transition
    _parameters = ['gamma', 'gamma_p', 'Delta']

    PORTSIN = ['PIn', 'FIn1', 'FIn2']
    PORTSOUT = ['POut']

    sub_blockstructure = (1, 1, 1, 1, 1)

    def _creduce(self):
        return ProbePort(self) + FeedbackPort(self, 1) + FeedbackPort(self, 2) + LossPort(self, 1) + LossPort(self, 2)

    def _toSLH(self):
        return self.creduce().toSLH()

    @property
    def space(self):
        return LocalSpace(self.name, basis=('r','h','g'))


class ProbePort(SubComponent):
    """
    Probe beam port for the Z-Probe cavity model.
    """

    def __init__(self, cavity):
        super(ProbePort, self).__init__(cavity, 0)

    def _toSLH(self):

        S = Matrix([[Z(self.space)]])
        L = Matrix([[0]])
        H = self.Delta * LocalProjector(self.space, 'r')

        return SLH(S, L, H)


class FeedbackPort(SubComponent):
    """
    Feedback beam port for the Z-Probe cavity model.
    """

    def _toSLH(self):

        S =  identity_matrix(1)

        if self.sub_index == 1:
            L = sqrt(self.gamma) * Matrix([[LocalSigma(self.space, 'g', 'r')]])
        elif self.sub_index == 2:
            L = sqrt(self.gamma) * Matrix([[LocalSigma(self.space, 'h', 'r')]])
        else:
            raise Exception(str(self.sub_index))

        return SLH(S, L, 0)


class LossPort(SubComponent):
    """
    Spontaneous decay from the far detuned excited r level.
    """

    def _toSLH(self):
        S = identity_matrix(1)
        if self.sub_index == 1:
            L = sqrt(self.gamma_p) * Matrix([[LocalSigma(self.space, 'g', 'r')]])
        elif self.sub_index == 2:
            L = sqrt(self.gamma_p) * Matrix([[LocalSigma(self.space, 'h', 'r')]])
        else:
            raise Exception(str(self.sub_index))

        return SLH(S, L, 0)
