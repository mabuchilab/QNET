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
Component definition file for a degenerate OPO model with three signal beam ports.
See documentation of :py:class:`ThreePortOPO`.
"""
import unittest

from qnet.algebra.circuit_algebra import HilbertSpace, Destroy, Matrix, sqrt, SLH, LocalSigma, identity_matrix, local_space
from qnet.circuit_components.component import Component, SubComponent

from sympy.core.symbol import symbols
from sympy import I

class ThreePortOPO(Component):
    r"""
    This model describes a degenerate OPO with three signal beam ports
    in the sub-threshold regime. I.e., the pump is modeled as a classical amplitude.

    The model's SLH parameters are given by

    .. math::
        S & = \mathbf{1}_3 \\
        L & = \begin{pmatrix} \sqrt{\kappa_1} a \\ \sqrt{\kappa_2} a \\ \sqrt{\kappa_3} a \end{pmatrix} \\
        H &= \Delta a^\dagger a + {i\over 2} \left( \alpha {a^\dagger}^2 - \alpha^* a^2\right)

    This particular component definition explicitly captures the reducibility of a trivial scattering matrix.
    I.e., it can be reduced into separate :py:class:`OPOPort` models for each port.


    Note that this model's validity breaks down even in open-loop configuration when

    .. math::
        |\alpha| > {\kappa_1 + \kappa_2 + \kappa_3\over 2}

    which is just the threshold condition.
    In a feedback configuration the threshold condition is generally changed.
    """
    
    CDIM = 3

    name = "OPO"

    kappa_1 = symbols('kappa_1', real = True) # coupling through first port
    kappa_2 = symbols('kappa_2', real = True) # coupling through second port
    kappa_3 = symbols('kappa_2', real = True) # coupling throug third port
    alpha = symbols('alpha')   # pump amplitude
    Delta = symbols('Delta', real = True) # detuning between the cavity (mode) and external driving field
    FOCK_DIM = 25
    _parameters = ['kappa_1', 'kappa_2','kappa_3', 'alpha', 'Delta', 'FOCK_DIM']

    
    PORTSIN = ['In1', 'In2', 'In3']
    PORTSOUT = ['Out1', 'Out2', 'In3']
    
    sub_blockstructure = (1, 1, 1)

    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)
    
    def _creduce(self):
        return OPOPort(self, 0) + OPOPort(self, 1) + OPOPort(self, 2)

    def _toSLH(self):
        return self.creduce().toSLH()
        

class OPOPort(SubComponent):
    """
    Sub component model for the individual ports of a :py:class:`ThreePortOPO`.
    The Hamiltonian is included with the first port.
    """

    def _toSLH(self):

        a = Destroy(self.space)
        a_d = a.adjoint()

        S = identity_matrix(1)

        if self.sub_index == 0: 
            # Include the Hamiltonian only with the first port of the kerr cavity circuit object
            H = self.Delta * a_d * a + (I/2) * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
            L = Matrix([[sqrt(self.kappa_1) * a]])
        elif self.sub_index == 1:
            H = 0
            L = Matrix([[sqrt(self.kappa_2) * a]])
        else:
            H = 0
            L = Matrix([[sqrt(self.kappa_3) * a]])

        return SLH(S, L, H)



# Test the circuit
class _TestThreePortOPO(unittest.TestCase):

    def testCreation(self):
        a = ThreePortOPO()
        self.assertIsInstance(a, ThreePortOPO)

    def testCReduce(self):
        a = ThreePortOPO().creduce()

    def testParameters(self):
        if len(ThreePortOPO._parameters):
            pname = ThreePortOPO._parameters[0]
            obj = ThreePortOPO(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = ThreePortOPO(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = ThreePortOPO().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()