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
Component definition file for a Kerr-nonlinear cavity model with two ports.
See documentation of :py:class:`TwoPortKerrCavity`.
"""
import unittest

from qnet.circuit_components.component import Component, SubComponent
from qnet.circuit_components.library import make_namespace_string
from qnet.algebra.circuit_algebra import HilbertSpace, Destroy, local_space, IdentityMatrix, Matrix, sqrt, SLH, tex, identity_matrix
from sympy.core.symbol import symbols




class TwoPortKerrCavity(Component):
    r"""
    This model describes a Kerr cavity model with two ports.

    The model's SLH parameters are given by

    .. math::
        S & = \mathbf{1}_2 \\
        L & = \begin{pmatrix} \sqrt{\kappa_1} a \\ \sqrt{\kappa_2} \end{pmatrix} \\
        H &= \Delta a^\dagger a + \chi {a^\dagger}^2 a^2

    This particular component definition explicitly captures the reducibility of a trivial scattering matrix.
    I.e., it can be reduced into separate :py:class:`KerrPort` models for each port.
    """
    
    CDIM = 2
    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']
    
    sub_blockstructure = (1, 1, 1)

    name = "K"
    namespace = ""

    Delta = symbols('Delta', real = True)       # Detuning from cavity
    chi = symbols('chi', real = True)           # Kerr-nonlinear coefficient
    kappa_1 = symbols('kappa_1', positive = True)   # coupling through first port
    kappa_2 = symbols('kappa_2', positive = True)   # coupling through first port
    FOCK_DIM = 75
    _parameters = ['Delta', 'chi', 'kappa_1', 'kappa_2', FOCK_DIM]



    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)

    @property
    def port1(self):
        return KerrPort(self, 0)

    @property
    def port2(self):
        return KerrPort(self, 1)
    
    def _creduce(self):
        return self.port1 + self.port2

    def _toSLH(self):
        return self.creduce().toSLH()

    def _toABCD(self, linearize):
        return self.toSLH().toABCD(linearize)
    


class KerrPort(SubComponent):
    """
    Sub component model for the individual ports of a :py:class:`TwoPortKerrCavity`.
    The Hamiltonian is included with the first port.
    """
    
    def _toSLH(self):

        a = Destroy(self.space)
        a_d = a.adjoint()
        S = identity_matrix(1)
        kappas = [self.kappa_1, self.kappa_2]
        kappa = kappas[self.sub_index]
        
        L = Matrix([[sqrt(kappa) * a]])
        # Include the Hamiltonian only with the first port of the kerr cavity circuit object
        H = self.Delta * (a_d * a) + self.chi * (a_d * a_d * a * a) if self.sub_index == 0 else 0
                
        return SLH(S, L, H)





# Test the circuit
class _TestTwoPortKerrCavity(unittest.TestCase):


  def testCreation(self):
      a = TwoPortKerrCavity()
      self.assertIsInstance(a, TwoPortKerrCavity)

  def testCReduce(self):
      a = TwoPortKerrCavity().creduce()

  def testParameters(self):
      if len(TwoPortKerrCavity._parameters):
          pname = TwoPortKerrCavity._parameters[0]
          obj = TwoPortKerrCavity(name="TestName", namespace="TestNamespace", **{pname: 5})
          self.assertEqual(getattr(obj, pname), 5)
          self.assertEqual(obj.name, "TestName")
          self.assertEqual(obj.namespace, "TestNamespace")

      else:
          obj = TwoPortKerrCavity(name="TestName", namespace="TestNamespace")
          self.assertEqual(obj.name, "TestName")
          self.assertEqual(obj.namespace, "TestNamespace")

  def testToSLH(self):
      aslh = TwoPortKerrCavity().toSLH()
      self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
  unittest.main()
