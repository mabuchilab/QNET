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
open_lossy_cc.py

Created automatically by $QNET/bin/parse_qhdl.py
Get started by instantiating a circuit instance via:

    ``OpenLossy()``

"""

__all__ = ['OpenLossy']

from qnet.circuit_components.library import make_namespace_string
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import cid, P_sigma, FB, SLH
import unittest
from sympy import symbols
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.kerr_cavity_cc import KerrCavity



class OpenLossy(Component):
    
    # total number of field channels
    CDIM = 3
    
    # parameters on which the model depends
    Delta = symbols('Delta', real = True)
    chi = symbols('chi', real = True)
    kappa = symbols('kappa', real = True)
    theta = symbols('theta', real = True)
    theta_LS0 = symbols('theta_LS0', real = True)
    _parameters = ['Delta', 'chi', 'kappa', 'theta', 'theta_LS0']

    # list of input port names
    PORTSIN = ['In1']
    
    # list of output port names
    PORTSOUT = ['Out1', 'Out2']

    # sub-components
    
    @property
    def BS(self):
        return Beamsplitter(make_namespace_string(self.name, 'BS'), theta = self.theta)

    @property
    def KC(self):
        return KerrCavity(make_namespace_string(self.name, 'KC'), kappa_2 = self.kappa, chi = self.chi, kappa_1 = self.kappa, Delta = self.Delta)

    @property
    def LSS_ci_ls(self):
        return Beamsplitter(make_namespace_string(self.name, 'LSS_ci_ls'), theta = self.theta_LS0)

    _sub_components = ['BS', 'KC', 'LSS_ci_ls']
    

    def _toSLH(self):
        return self.creduce().toSLH()
        
    def _creduce(self):

        BS, KC, LSS_ci_ls = self.BS, self.KC, self.LSS_ci_ls

        return (KC + cid(1)) << P_sigma(0, 2, 1) << (LSS_ci_ls + cid(1)) << P_sigma(0, 2, 1) << (BS + cid(1))

    @property
    def _space(self):
        return self.creduce().space


# Test the circuit
class TestOpenLossy(unittest.TestCase):
    """
    Automatically created unittest test case for OpenLossy.
    """

    def testCreation(self):
        a = OpenLossy()
        self.assertIsInstance(a, OpenLossy)

    def testCReduce(self):
        a = OpenLossy().creduce()

    def testParameters(self):
        if len(OpenLossy._parameters):
            pname = OpenLossy._parameters[0]
            obj = OpenLossy(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = OpenLossy(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = OpenLossy().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()