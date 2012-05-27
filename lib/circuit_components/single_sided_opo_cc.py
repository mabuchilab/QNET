#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from algebra.circuit_algebra import HilbertSpace, Destroy, SpaceExists, OperatorMatrixInstance, sqrt, SLH, LocalSigma
from library import make_namespace_string
from circuit_components.component import Component, SubComponent

from sympy.core.symbol import symbols
from sympy import I

class SingleSidedOPO(Component):
    """
    comment
    """
    
    CDIM = 1
    GENERIC_DEFAULT_VALUES = dict(
            kappa = symbols('kappa', real = True, each_char = False), # decay of cavity mode through cavity mirror
            alpha = symbols('alpha', each_char = False),   # coupling between cavity mode and two-level-system
            Delta = symbols('Delta', real = True, each_char = False), # detuning between the cavity (mode) and the atomic transition
            FOCK_DIM = 25,
            )
    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']        
        
    # def reduce(self):
    #     return self
        
    def toSLH(self):
        try:
            #hilbert space for two-level-system
            self.s_id = HilbertSpace.register_local_space(self.name, range(self.FOCK_DIM))
        except SpaceExists:
            self.s_id = HilbertSpace.retrieve_by_descriptor(self.name)[0]
            HilbertSpace.set_states(self.name, range(self.FOCK_DIM))

        a = Destroy(self.s_id)
        a_d = a.adjoint()

        #coupling to external mode
        L = sqrt(self.kappa) * a

        H = self.Delta * a_d * a + I * (self.alpha * a_d * a_d - self.alpha.conjugate() * a * a)
        return SLH(OperatorMatrixInstance([[1]]),OperatorMatrixInstance([[L]]), H)
 

def test():
    a = SingleSidedOPO(Delta = symbols("MyDelta", real = True))
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
