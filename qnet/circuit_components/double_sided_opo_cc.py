#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from algebra.circuit_algebra import HilbertSpace, Destroy, SpaceExists, OperatorMatrixInstance, sqrt, SLH, LocalSigma, IdentityMatrix
from library import make_namespace_string
from circuit_components.component import Component, SubComponent

from sympy.core.symbol import symbols
from sympy import I

class DoubleSidedOPO(Component):
    """
    comment
    """
    
    CDIM = 2
    GENERIC_DEFAULT_VALUES = dict(
            kappa_1 = symbols('kappa_1', real = True, each_char = False), # decay of cavity mode through cavity mirror
            kappa_2 = symbols('kappa_2', real = True, each_char = False), # decay rate into transverse modes
            alpha = symbols('alpha', each_char = False),   # coupling between cavity mode and two-level-system
            Delta = symbols('Delta', real = True, each_char = False), # detuning between the cavity (mode) and the atomic transition
            FOCK_DIM = 25,
            )
    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']        
    
    sub_blockstructure = (1, 1)
    
    def reduce(self):        
        try:
            self.fock_id = HilbertSpace.register_local_space(self.name, range(self.FOCK_DIM))
        except SpaceExists:
            self.fock_id = HilbertSpace.retrieve_by_descriptor(self.name)[0]
            HilbertSpace.set_states(self.name, range(self.FOCK_DIM))
            
        return OPOPort(self, 0) + OPOPort(self, 1)
        
    def toSLH(self):
        return self.reduce().toSLH()
        

class OPOPort(SubComponent):

    def toSLH(self):

        a = Destroy(self.fock_id)
        a_d = a.adjoint()
        S = IdentityMatrix(1)

        if self.sub_index == 0: 
            # Include the Hamiltonian only with the first port of the kerr cavity circuit object
            H = self.Delta * a_d * a + I * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
            L = OperatorMatrixInstance([[sqrt(self.kappa_1) * a]])
        else:
            H = 0
            L = OperatorMatrixInstance([[sqrt(self.kappa_2) * a]])

        return SLH(S, L, H)



def test():
    a = DoubleSidedOPO(Delta = symbols("MyDelta", real = True))
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
