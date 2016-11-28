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
Component definition file for a degenerate OPO model with a single port for the signal beam.
See documentation of :py:class:`SingleSidedOPO`.
"""
from sympy.core.symbol import symbols
from sympy import I, sqrt

from qnet.algebra.circuit_algebra import SLH
from qnet.algebra.matrix_algebra import Matrix, identity_matrix
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import Destroy
from qnet.circuit_components.component import Component


class SingleSidedOPO(Component):
    r"""This model describes a degenerate OPO with a single port for the signal
    mode in the sub-threshold regime: i.e., the pump is modeled as a classical
    amplitude.

    The model's SLH parameters are given by

    .. math::
        S & = (1) \\
        L & = \begin{pmatrix} \sqrt{\kappa} a  \end{pmatrix} \\
        H &= \Delta a^\dagger a + {i\over 2} \left( \alpha {a^\dagger}^2 - \alpha^* a^2\right)

    """

    CDIM = 1

    kappa = symbols('kappa', real = True) # decay of cavity mode through cavity mirror
    alpha = symbols('alpha')   # coupling between cavity mode and two-level-system
    Delta = symbols('Delta', real = True) # detuning between the cavity (mode) and the atomic transition
    FOCK_DIM = 25
    _parameters = ['kappa', 'alpha', 'Delta', 'FOCK_DIM']


    PORTSIN = ['In1']
    PORTSOUT = ['Out1']


    @property
    def space(self):
        return LocalSpace(self.name, dimension=self.FOCK_DIM)

    def _toSLH(self):

        a = Destroy(hs=self.space)
        a_d = a.adjoint()

        S = identity_matrix(1)


        # Include the Hamiltonian only with the first port of the kerr cavity circuit object
        H = self.Delta * a_d * a + (I / 2) * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
        L = Matrix([[sqrt(self.kappa) * a]])

        return SLH(S, L, H)
