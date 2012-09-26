#!/usr/bin/env python
# encoding: utf-8
"""
Component definition file for a coherent field displacement component.
See documentation of :py:class:`Displace`.
"""



import unittest
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import *
from sympy.core.symbol import symbols



class Displace(Component):
    r"""
    Coherent displacement of the input field (usually vacuum) by a complex amplitude :math:`\alpha`.
    This component serves as the model of an ideal laser source without internal non-classical internal dynamics.
    """
    
    CDIM = 1
    

    alpha = symbols('alpha') # complex valued laser amplitude
    _parameters = ['alpha']

    
    PORTSIN = ["VacIn"]
    PORTSOUT = ["Out1"]
    
    def _toSLH(self):
        
        S = Matrix([[1]])
        L = Matrix([[self.alpha]])
        H = 0
        
        return SLH(S, L, H)

    _space = TrivialSpace
        
    def _tex(self):
        return r"{W(%s)}" % tex(self.alpha)




# Test the circuit
class _TestDisplace(unittest.TestCase):

    def testCreation(self):
        a = Displace()
        self.assertIsInstance(a, Displace)

    def testCReduce(self):
        a = Displace().creduce()

    def testParameters(self):
        if len(Displace._parameters):
            pname = Displace._parameters[0]
            obj = Displace(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = Displace(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = Displace().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()