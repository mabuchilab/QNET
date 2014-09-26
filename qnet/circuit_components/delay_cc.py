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

This components takes parameters tau (delay length), omega_max (max frequency), and 
err (desired error). These are used to compute the number of cavities (N).

Alternatively, the user may give N and kappa, which is used as a parameter for all cavities.
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
    
    PORTSIN = ["In1"]
    PORTSOUT = ["Out1"]

    name = "T"
    namespace = ""

    # positive valued delay, maximum frequency, and allowed error
    tau = symbols('tau', positive = True)
    omega_max = symbols('omega_max', positive = True)
    err = symbols('err', positive = True)
    kappa = symbols('kappa', positive = True)
    N = symbols('N', positive = True)
    Delta = symbols('Delta', positive = True)
    
    _parameters = ['tau', 'omega_max', 'err', 'kappa', 'N', 'Delta']

    # takes the relative error required.
    # returns [kappa, L] for a single cavity, optimized with T = 1.
    def _cavity_values(self,err):
        err = err**2
        err_vals = [1.1848189096497208e-11,
                    1.1539713629105108e-10,
                    1.1811122080374048e-09,
                    1.0029879926598539e-08,
                    1.0310528963941579e-07,
                    1.0561655805485515e-06,
                    1.0144204457773753e-05,
                    0.0001016636,
                    0.0010103962,
                    0.0050077302,
                    0.0100810559]
        
        returns = [[3.99915799, 0.0648897796],
                   [3.9982020452,0.0948296593],
                   [3.9960972991,0.139739479],
                   [3.9920417727,0.1996192385],
                   [3.9827159963,0.2944288577],
                   [3.9625566947,0.4341482966],
                   [3.9208237531,0.633747495],
                   [3.8312474713,0.9331462926],
                   [3.6461141479,1.377254509],
                   [3.4164640904,1.8163727455],
                   [3.2789417079,2.0558917836] ]
        index = 0;
        for i in err_vals:
            if err > i:
                index += 1
        return returns[index] 

    
    #single cavity with zero detuning
    def _SLH_Cav(self, kappa = 0, Delta = 0, S_num = 1, namespace = ''):
        
        fock = local_space('fock', namespace = namespace)
        
        # create representations of a and sigma
        a = Destroy(fock)
        
        # Trivial scattering matrix
        S = identity_matrix(1) * S_num
        
        L = Matrix([[ sqrt(kappa) * a]])
        
        # Hamilton operator
        H = Delta * a.dag()*a
        
        return SLH(S, L, H) 
    
    def _toSLH(self):
        return self._creduce().toSLH()

    ##maybe do it differently so you can access _sub_components ?
        
    def _creduce(self):
    
    	if type(self.Delta) != float:
    	    self.Delta = 0
    
    	if type(self.N) != int:

            #compute number of cavities and kappa for each given 
            #tau, omega_max, and err.
            [kappa_single, product_single] = self._cavity_values(self.err)
            self.N = (int) (self.tau * self.omega_max / product_single) + 1
            self.kappa = kappa_single * self.N / self.tau
        
        #TD = self._SLH_Cav(kappa = self.kappa, namespace = '%s_%s' % (self.namespace,0)
        TD = SLH(identity_matrix(1),Matrix([[0]]),0)
        for i in range(0,self.N):
            cav = self._SLH_Cav(kappa = self.kappa, namespace = '%s_%s' % (self.namespace, i) )
            TD = TD << cav
        return TD


    @property
    def _space(self):
        return self.creduce().space

    @property
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
        print aslh
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()
