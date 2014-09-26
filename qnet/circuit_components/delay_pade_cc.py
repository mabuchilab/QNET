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

This components takes parameters tau (delay length) and N.  A Pade approximation is
used to construct a time delay. For long delays (N > 50) there may be issues because
a large factorial is taken. To avoid this, you can concatenate several time delays
or replace ints and floats with higher precision data types.
"""



import unittest
import numpy as np
import sympy as sym
import math


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

    # denominator of symmetric Pade approximation of e^{-s} of order n
    def pade_approx(self, n):
        output = [0]*(n+1)
        for k in xrange(0,n+1):
            output[n-k] = ( (float(math.factorial(2*n-k))) * math.factorial(n) / \
                          ((math.factorial(2*n) )* math.factorial(k) * math.factorial(n - k) ) )
        return output

    def pade_roots(self,n):
        return np.roots(self.pade_approx(n))
        
    #uses the pade roots to extract cavity delta and kappa
    #notice this is for delay of length T = 1.
    #for general T, divide each k by T.
    def ks_ds(self, n):
        roots = self.pade_roots(n)
        ks = 2.*np.copy(-roots.real)
        ds = np.copy(roots.imag)
        #Protect against division by zero, which may happen later:
        for i in xrange(0,len(ds)):
            if abs(ds[i]) < 1e-30 :
                ds[i] = 1e-30
        return [ks,ds]
    
        
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
        
        [ks,ds] = self.ks_ds(self.N)
        
        ks /= self.tau  #adjust the interaction terms for the right delay time.
        ds /= self.tau
        
        TD = SLH(identity_matrix(1),Matrix([[0]]),0) #NO Hamiltonian.
        for i in range(0,self.N):
            cav = self._SLH_Cav(kappa = ks[i], Delta = ds[i],
            	                S_num = 1, namespace = '%s_%s' % (self.namespace, i) )
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
