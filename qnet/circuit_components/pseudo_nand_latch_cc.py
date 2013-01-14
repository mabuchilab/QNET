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
pseudo_nand_latch_cc.py

Created automatically by $QNET/bin/parse_qhdl.py
Get started by instantiating a circuit instance via:

    >>> PseudoNANDLatch()

"""

__all__ = ['PseudoNANDLatch']

from qnet.circuit_components.library import make_namespace_string
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import cid, P_sigma, FB, SLH
import unittest
from sympy import symbols
from qnet.circuit_components.pseudo_nand_cc import PseudoNAND



class PseudoNANDLatch(Component):

    # total number of field channels
    CDIM = 6
    
    # parameters on which the model depends
    
    _parameters = []

    # list of input port names
    PORTSIN = ['NS', 'W1', 'kerr2_extra', 'NR', 'W2', 'kerr1_extra']
    
    # list of output port names
    PORTSOUT = ['BS1_1_out', 'kerr1_out2', 'OUT2_2', 'BS1_2_out', 'kerr2_out2', 'OUT2_1']

    # sub-components
    
    @property
    def NAND1(self):
        return PseudoNAND(make_namespace_string(self.name, 'NAND1'))

    @property
    def NAND2(self):
        return PseudoNAND(make_namespace_string(self.name, 'NAND2'))

    _sub_components = ['NAND1', 'NAND2']
    

    def _toSLH(self):
        return self.creduce().toSLH()
        
    def _creduce(self):

        NAND1, NAND2 = self.NAND1, self.NAND2

        return P_sigma(3, 4, 2, 0, 1, 5) << FB(((P_sigma(0, 1, 3, 2) << NAND2) + cid(3)) << (cid(1) + (P_sigma(0, 3, 4, 5, 1, 2) << ((P_sigma(1, 2, 0, 3) << NAND1) + cid(2)))), 3, 2) << P_sigma(1, 2, 5, 0, 4, 3)

    @property
    def _space(self):
        return self.creduce().space


# Test the circuit
class TestPseudoNANDLatch(unittest.TestCase):
    """
    Automatically created unittest test case for PseudoNANDLatch.
    """

    def testCreation(self):
        a = PseudoNANDLatch()
        self.assertIsInstance(a, PseudoNANDLatch)

    def testCReduce(self):
        a = PseudoNANDLatch().creduce()

    def testParameters(self):
        if len(PseudoNANDLatch._parameters):
            pname = PseudoNANDLatch._parameters[0]
            obj = PseudoNANDLatch(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = PseudoNANDLatch(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = PseudoNANDLatch().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()