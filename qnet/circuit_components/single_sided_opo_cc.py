#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.algebra.circuit_algebra import Destroy, local_space, Matrix, sqrt, SLH
from qnet.circuit_components.component import Component

from sympy.core.symbol import symbols
from sympy import I

class SingleSidedOPO(Component):
    """
    comment
    """
    
    CDIM = 1
    
    kappa = symbols('kappa', real = True) # decay of cavity mode through cavity mirror
    alpha = symbols('alpha')              # coupling between cavity mode and two-level-system
    Delta = symbols('Delta', real = True) # detuning between the cavity (mode) and the atomic transition
    FOCK_DIM = 25
    _parameters = ['kappa', 'alpha', 'Delta', 'FOCK_DIM']

    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']        


    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)

    def _toSLH(self):
        a = Destroy(self.space)
        a_d = a.adjoint()

        #coupling to external mode
        L = sqrt(self.kappa) * a

        H = self.Delta * a_d * a + I * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
        return SLH(Matrix([[1]]),Matrix([[L]]), H)
 

def test():
    a = SingleSidedOPO()
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
