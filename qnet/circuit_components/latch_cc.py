#!/usr/bin/env python
# encoding: utf-8
#This file was automatically created using QNET.

"""
latch_cc.py

Created automatically by $QNET/bin/parse_qhdl.py
Get started by instantiating a circuit instance via:

    >>> Latch()

"""

__all__ = ['Latch']

from qnet.circuit_components.library import make_namespace_string
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import cid, P_sigma, FB, SLH
import unittest
from sympy import symbols
from qnet.circuit_components.three_port_kerr_cavity_cc import ThreePortKerrCavity
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.phase_cc import Phase
from qnet.circuit_components.displace_cc import Displace



class Latch(Component):

    # total number of field channels
    CDIM = 8
    
    # parameters on which the model depends
    Delta = 50.0
    chi = -0.205
    kappa_1 = 20.0
    kappa_2 = 20.0
    kappa_3 = 10.0
    theta = 0.891
    thetap = 0.593
    phi = 2.72
    phip = 0.14
    beta = (-79.838356622-35.806239846j)
    _parameters = ['Delta', 'beta', 'chi', 'kappa_1', 'kappa_2', 'kappa_3', 'phi', 'phip', 'theta', 'thetap']

    # list of input port names
    PORTSIN = ['In1', 'In2']
    
    # list of output port names
    PORTSOUT = ['Out1']

    # sub-components
    
    @property
    def B11(self):
        return Beamsplitter(make_namespace_string(self.name, 'B11'))

    @property
    def B12(self):
        return Beamsplitter(make_namespace_string(self.name, 'B12'), theta = self.theta)

    @property
    def B21(self):
        return Beamsplitter(make_namespace_string(self.name, 'B21'))

    @property
    def B22(self):
        return Beamsplitter(make_namespace_string(self.name, 'B22'), theta = self.theta)

    @property
    def B3(self):
        return Beamsplitter(make_namespace_string(self.name, 'B3'), theta = self.thetap)

    @property
    def C1(self):
        return ThreePortKerrCavity(make_namespace_string(self.name, 'C1'), kappa_2 = self.kappa_2, chi = self.chi, kappa_1 = self.kappa_1, kappa_3 = self.kappa_3, Delta = self.Delta)

    @property
    def C2(self):
        return ThreePortKerrCavity(make_namespace_string(self.name, 'C2'), kappa_2 = self.kappa_2, chi = self.chi, kappa_1 = self.kappa_1, kappa_3 = self.kappa_3, Delta = self.Delta)

    @property
    def Phase1(self):
        return Phase(make_namespace_string(self.name, 'Phase1'), phi = self.phi)

    @property
    def Phase2(self):
        return Phase(make_namespace_string(self.name, 'Phase2'), phi = self.phi)

    @property
    def Phase3(self):
        return Phase(make_namespace_string(self.name, 'Phase3'), phi = self.phip)

    @property
    def W1(self):
        return Displace(make_namespace_string(self.name, 'W1'), alpha = self.beta)

    @property
    def W2(self):
        return Displace(make_namespace_string(self.name, 'W2'), alpha = self.beta)

    _sub_components = ['B11', 'B12', 'B21', 'B22', 'B3', 'C1', 'C2', 'Phase1', 'Phase2', 'Phase3', 'W1', 'W2']
    

    def _toSLH(self):
        return self.creduce().toSLH()
        
    def _creduce(self):

        B11, B12, B21, B22, B3, C1, C2, Phase1, Phase2, Phase3, W1, W2 = self.B11, self.B12, self.B21, self.B22, self.B3, self.C1, self.C2, self.Phase1, self.Phase2, self.Phase3, self.W1, self.W2

        return P_sigma(1, 2, 3, 4, 5, 6, 7, 0) << FB((cid(4) + (P_sigma(0, 4, 1, 2, 3) << (B11 + cid(3)))) << P_sigma(0, 1, 2, 3, 4, 6, 7, 8, 5) << ((P_sigma(0, 1, 2, 3, 4, 7, 5, 6) << ((P_sigma(0, 1, 5, 3, 4, 2) << FB((cid(2) + ((B3 + cid(1)) << P_sigma(0, 2, 1) << (B12 + cid(1))) + cid(2)) << P_sigma(0, 1, 4, 5, 6, 2, 3) << (((cid(1) + ((cid(1) + ((Phase3 + Phase2) << B22) + cid(1)) << P_sigma(0, 1, 3, 2) << (C2 + W2))) << ((B21 << (Phase1 + cid(1))) + cid(3))) + cid(2)), 4, 0)) + cid(2)) << (cid(4) + (P_sigma(0, 2, 3, 1) << ((P_sigma(1, 0, 2) << C1) + W1)))) + cid(1)), 8, 4) << P_sigma(7, 0, 6, 3, 1, 2, 4, 5)

    @property
    def _space(self):
        return self.creduce().space


# Test the circuit
class TestLatch(unittest.TestCase):
    """
    Automatically created unittest test case for Latch.
    """

    def testCreation(self):
        a = Latch()
        self.assertIsInstance(a, Latch)

    def testCReduce(self):
        a = Latch().creduce()

    def testParameters(self):
        if len(Latch._parameters):
            pname = Latch._parameters[0]
            obj = Latch(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = Latch(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = Latch().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()