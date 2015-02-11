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
Component definition file for a pseudo-delay model that works over a limited bandwidth.
See documentation of :py:class:`Delay`.
"""



import unittest
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import *
from sympy.core.symbol import symbols
from numpy import array as np_array
from qnet.circuit_components.library import make_namespace_string
from functools import reduce as freduce


class Delay(Component):
    r"""
    Delay
    """
    
    CDIM = 1
    
    name = "T"
    namespace = ""

    tau = symbols('tau', positive = True) # positive valued delay
    N = 3
    FOCK_DIM = 25

    _parameters = ['alpha', 'N', 'FOCK_DIM']

    
    PORTSIN = ["In1"]
    PORTSOUT = ["Out1"]
    
    def _toSLH(self):

        # These numerically optimal solutions were obtained as outlined in
        # my blog post on the Mabuchi-Lab internal blog
        # email me (ntezak@stanford.edu)for details.
        if self.N == 1:
            kappa0 = 9.28874141848 / self.tau
            kappas = np_array([7.35562929]) / self.tau
            Deltas = np_array([3.50876192]) / self.tau
        elif self.N == 3:
            kappa0 = 14.5869543803 / self.tau
            kappas = np_array([ 13.40782559, 9.29869721]) / self.tau
            Deltas = np_array([3.48532283, 7.14204585]) / self.tau
        elif self.N == 5:
            kappa0 = 19.8871474779 / self.tau
            kappas = np_array([19.03316217, 10.74270752, 16.28055664]) / self.tau
            Deltas = np_array([3.47857213, 10.84138821, 7.03434809]) / self.tau
        else:
            raise NotImplementedError("The number of cavities to realize the delay must be one of 1,3 or 5.")

        h0 = make_namespace_string(self.name, 'C0')
        hp =  [make_namespace_string(self.name, "C{:d}p".format(n+1)) for n in range((self.N-1)/2)]
        hm =  [make_namespace_string(self.name, "C{:d}m".format(n+1)) for n in range((self.N-1)/2)]


        S = Matrix([1.])
        slh0 = SLH(S, Matrix([[sqrt(kappa0) * Destroy(h0)]]), ZeroOperator)
        slhp = [SLH(S, Matrix([[sqrt(kj) * Destroy(hj)]]), Dj * Create(hj) * Destroy(hj)) for (kj, Dj, hj) in zip(kappas, Deltas, hp)]
        slhm = [SLH(S, Matrix([[sqrt(kj) * Destroy(hj)]]), -Dj * Create(hj) * Destroy(hj)) for (kj, Dj, hj) in zip(kappas, Deltas, hm)]

        return freduce(lambda a, b: a << b, slhp + slhm, slh0)



    _space = TrivialSpace
        
    def _tex(self):
        return r"{T(%s)}" % tex(self.tau)




# Test the circuit
class _TestDelay(unittest.TestCase):

    def testCreation(self):
        a = Delay()
        self.assertIsInstance(a, Delay)

    def testCReduce(self):
        a = Delay().creduce()

    def testParameters(self):
        if len(Delay._parameters):
            pname = Delay._parameters[0]
            obj = Delay(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = Delay(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = Delay().toSLH()
        print(aslh)
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()