#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.circuit_components.component import Component, SubComponent

from qnet.algebra.circuit_algebra import Destroy, identity_matrix, Matrix, sqrt, SLH, local_space
from sympy.core.symbol import symbols



class LinearCavity(Component):
    
    CDIM = 2
    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']
    

    Delta = symbols('Delta', real = True)          # Detuning from cavity
    kappa_1 = symbols('kappa_1', real = True)      # coupling through first port
    kappa_2 = symbols('kappa_2', real = True)      # coupling through second port
    FOCK_DIM = 75                                  # Truncated Fock dimension
    _parameters = ['Delta', 'kappa_1', 'kappa_2', FOCK_DIM]
    
    sub_blockstructure = (1, 1)

    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)
        
    def _creduce(self):
        return LinearCavityPort(self, 0) + LinearCavityPort(self, 1)
    
    def _toSLH(self):
        return self.creduce().toSLH()


class LinearCavityPort(SubComponent):
    
    def _toSLH(self):
        
        a = Destroy(self.space)
        a_d = a.adjoint()
        S = identity_matrix(1)
        
        if self.sub_index == 0: 
            # Include the Hamiltonian only with the first port of the kerr cavity circuit object
            H = self.Delta * (a_d * a)
            L = Matrix([[sqrt(self.kappa_1) * a]])
        else:
            H = 0
            L = Matrix([[sqrt(self.kappa_2) * a]])
        
        return SLH(S, L, H)



if __name__ == '__main__':
    a = LinearCavity()
    print a
    sa = a.creduce()
    print "-"*30
    print sa.__repr__()
    print "-"*30
    print sa.toSLH()

    
