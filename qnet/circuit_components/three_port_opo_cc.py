#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.algebra.circuit_algebra import Destroy, Matrix, sqrt, SLH, local_space
from qnet.circuit_components.component import Component
from sympy.core.symbol import symbols
from sympy import I

class ThreePortOPO(Component):
    """
    Three port OPO model.
    """
    
    CDIM = 3


    _params = ['kappa_1', 'kappa_2', 'kappa_3', 'alpha', 'Delta', 'FOCK_DIM']

    kappa_1 = symbols('kappa_1', real = True)   # decay of cavity mode through cavity mirror
    kappa_2 = symbols('kappa_2', real = True)   # decay rate into transverse modes
    kappa_3 = symbols('kappa_3', real = True)   # decay rate into transverse modes
    alpha = symbols('alpha')                    # coupling between cavity mode and two-level-system
    Delta = symbols('Delta', real = True)       # detuning between the cavity (mode) and the atomic transition
    FOCK_DIM = 25                               # truncated Fock basis dimension
    _parameters = ['kappa_1', 'kappa_2', 'kappa_3', 'alpha', 'Delta', 'FOCK_DIM']

    
    PORTSIN = ['In1', 'In2', 'In3']
    PORTSOUT = ['Out1', 'Out2', 'Out3']        
    
    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)
        
    def _toSLH(self):
        a = Destroy(self.space)
        a_d = a.adjoint()

        #coupling to external mode
        L1 = sqrt(self.kappa_1) * a
        L2 = sqrt(self.kappa_2) * a
        L3 = sqrt(self.kappa_3) * a

        H = self.Delta * a_d * a + I * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
        return SLH(Matrix([[1,0,0],[0,1,0],[0,0,1]]),Matrix([[L1],[L2],[L3]]), H)

 

def test():
    a = ThreePortOPO()
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
