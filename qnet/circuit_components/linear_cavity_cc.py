#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from circuit_components.component import Component, SubComponent

from algebra.circuit_algebra import HilbertSpace, Destroy, SpaceExists, IdentityMatrix, OperatorMatrixInstance, sqrt, SLH
from sympy.core.symbol import symbols



class LinearCavity(Component):
    
    CDIM = 2
    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']
    
    GENERIC_DEFAULT_VALUES = dict(
        Delta = symbols('Delta', real = True, each_char = False),          # Detuning from cavity
        kappa_1 = symbols('kappa_1', real = True, each_char = False),      # coupling through first port
        kappa_2 = symbols('kappa_2', real = True, each_char = False),      # coupling through second port
        FOCK_DIM = 75,                                                     # Truncated Fock dimension
        )
    
    sub_blockstructure = (1, 1)
        
    def reduce(self):
        try:
            self.fock_id = HilbertSpace.register_local_space(self.name, range(self.FOCK_DIM))
        except SpaceExists:
            self.fock_id = HilbertSpace.retrieve_by_descriptor(self.name)[0]
            HilbertSpace.set_states(self.name, range(self.FOCK_DIM))

        return LinearCavityPort(self, 0) + LinearCavityPort(self, 1)
    
    def toSLH(self):
        return self.reduce().toSLH()
        

class LinearCavityPort(SubComponent):
    
    def toSLH(self):
        
        a = Destroy(self.fock_id)
        a_d = a.adjoint()
        S = IdentityMatrix(1)
        
        if self.sub_index == 0: 
            # Include the Hamiltonian only with the first port of the kerr cavity circuit object
            H = self.Delta * (a_d * a)
            L = OperatorMatrixInstance([[sqrt(self.kappa_1) * a]])
        else:
            H = 0
            L = OperatorMatrixInstance([[sqrt(self.kappa_2) * a]])
        
        return SLH(S, L, H)



if __name__ == '__main__':
    a = LinearCavity()
    print a
    sa = a.reduce()
    print "-"*30
    print sa.__repr__()
    print "-"*30
    print sa.toSLH()

    
