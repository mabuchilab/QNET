#!/usr/bin/env python
# encoding: utf-8
"""
Component definition file for a coherent field Phasement component.
See documentation of :py:class:`Phase`.
"""
import unittest

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import Matrix, exp, SLH, I, TrivialSpace
from sympy.core.symbol import symbols


class Phase(Component):
    r"""
    Coherent phase shift of the field passing through by real angle :math:`\phi`.
    """

    CDIM = 1

    phi = symbols('phi', real = True)    # Phase angle
    _parameters = ['phi']

    PORTSIN = ["In1"]
    PORTSOUT = ["Out1"]
    
    def _toSLH(self):
        
        S = Matrix([[exp(I * self.phi)]])
        L = Matrix([[0]])
        H = 0
        return SLH(S, L, H)

    def _toABCD(self, linearize):
        return self.toSLH().toABCD(linearize)

    _space = TrivialSpace

    def _creduce(self):
        return self
    



# Test the circuit
class _TestPhase(unittest.TestCase):

  def testCreation(self):
      a = Phase()
      self.assertIsInstance(a, Phase)

  def testCReduce(self):
      a = Phase().creduce()

  def testParameters(self):
      if len(Phase._parameters):
          pname = Phase._parameters[0]
          obj = Phase(name="TestName", namespace="TestNamespace", **{pname: 5})
          self.assertEqual(getattr(obj, pname), 5)
          self.assertEqual(obj.name, "TestName")
          self.assertEqual(obj.namespace, "TestNamespace")

      else:
          obj = Phase(name="TestName", namespace="TestNamespace")
          self.assertEqual(obj.name, "TestName")
          self.assertEqual(obj.namespace, "TestNamespace")

  def testToSLH(self):
      aslh = Phase().toSLH()
      self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
  unittest.main()