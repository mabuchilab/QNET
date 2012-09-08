#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.algebra.circuit_algebra import HilbertSpace, Destroy, Matrix, sqrt, SLH, LocalSigma, identity_matrix, local_space
from qnet.circuit_components.component import Component, SubComponent

from sympy.core.symbol import symbols
from sympy import I

class DoubleSidedOPO(Component):
    """
    comment
    """
    
    CDIM = 2

    name = "OPO"

    kappa_1 = symbols('kappa_1', real = True) # decay of cavity mode through cavity mirror
    kappa_2 = symbols('kappa_2', real = True) # decay rate into transverse modes
    alpha = symbols('alpha')   # coupling between cavity mode and two-level-system
    Delta = symbols('Delta', real = True) # detuning between the cavity (mode) and the atomic transition
    FOCK_DIM = 25
    _parameters = ['kappa_1', 'kappa_2', 'alpha', 'Delta', 'FOCK_DIM']

    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']        
    
    sub_blockstructure = (1, 1)

    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)
    
    def _creduce(self):
        return OPOPort(self, 0) + OPOPort(self, 1)

    def _toSLH(self):
        return self.creduce().toSLH()
        

class OPOPort(SubComponent):

    def _toSLH(self):

        a = Destroy(self.space)
        a_d = a.adjoint()

        S = identity_matrix(1)

        if self.sub_index == 0: 
            # Include the Hamiltonian only with the first port of the kerr cavity circuit object
            H = self.Delta * a_d * a + I * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
            L = Matrix([[sqrt(self.kappa_1) * a]])
        else:
            H = 0
            L = Matrix([[sqrt(self.kappa_2) * a]])

        return SLH(S, L, H)



def test():
    a = DoubleSidedOPO(Delta = symbols("MyDelta", real = True))
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
