#!/usr/bin/env python
# encoding: utf-8
"""
Component definition file for the Z-probe cavity model of our QEC papers.
See documentation of :py:class:`ZProbeCavity`.
"""
import unittest

from qnet.algebra.circuit_algebra import (HilbertSpace, local_space, SLH, Matrix,
                                    LocalProjector, Z, sqrt, identity_matrix, LocalSigma)
from qnet.circuit_components.component import Component, SubComponent
from sympy.core.symbol import symbols


class ZProbeCavity(Component):
    """
    This is the Z-probe cavity model as used in our group's QEC papers [#qec1,#qec2]_ , which has three (dressed) internal
    states: r, g, h. The qubit is encoded in g,h, while r is used to drive transitions.
    The first channel is the probe-signal, while the second and third channels are 
    the two independent feedback beams.


    Since the scattering matrix is diagonal we provide sub component models
    for the individual subsystems: One :py:class:`ProbePort` and two :py:class:`FeedbackPort` instances..

    

    """
    CDIM = 3

    gamma = symbols('gamma', real = True)   # decay rate into transverse modes
    Delta = symbols('Delta', real = True)   # detuning between the cavity (mode) and the atomic transition
    _parameters = ['gamma', 'Delta']
            
    PORTSIN = ['PIn', 'FIn1', 'FIn2']
    PORTSOUT = ['POut', 'UOut1', 'UOut2']
    
    sub_blockstructure = (1, 1, 1)
    
    def _creduce(self):
        return ProbePort(self) + FeedbackPort(self, 1) + FeedbackPort(self, 2)

    def _toSLH(self):
        return self.creduce().toSLH()

    @property
    def _space(self):
        return local_space(self.name, self.namespace, basis = ('r','h','g'))



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


# Test the circuit
class _TestZProbeCavity(unittest.TestCase):
    def testCreation(self):
        a = ZProbeCavity()
        self.assertIsInstance(a, ZProbeCavity)

    def testCReduce(self):
        a = ZProbeCavity().creduce()

    def testParameters(self):
        if len(ZProbeCavity._parameters):
            pname = ZProbeCavity._parameters[0]
            obj = ZProbeCavity(name = "TestName", namespace = "TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = ZProbeCavity(name = "TestName", namespace = "TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = ZProbeCavity().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()