#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from algebra.circuit_algebra import HilbertSpace, Destroy, SpaceExists, OperatorMatrixInstance, sqrt, SLH, LocalSigma
from library import make_namespace_string
from component import Component, SubComponent

from sympy.core.symbol import symbols


class SingleSidedJaynesCummings(Component):
    """
    Typical CQED Jaynes-Cummings model with a single input port with coupling coefficient kappa
    and a single atomic excitation decay 'port' with rate gamma.
    """
    
    CDIM = 2
    GENERIC_DEFAULT_VALUES = dict(
            kappa = symbols('kappa', real = True, each_char = False), # decay of cavity mode through cavity mirror
            gamma = symbols('gamma', real = True, each_char = False), # decay rate into transverse modes
            g_c = symbols('g_c', real = True, each_char = False),   # coupling between cavity mode and two-level-system
            Delta = symbols('Delta', real = True, each_char = False), # detuning between the cavity (mode) and the atomic transition
            FOCK_DIM = 20,
            )
    
    PORTSIN = ['In1', 'VacIn']
    PORTSOUT = ['Out1', 'UOut']        
    
    sub_blockstructure = (1, 1)
    
    def reduce(self):
        try:
            #hilbert space for two-level-system
            self.tls_id = HilbertSpace.register_local_space(make_namespace_string(self.name,'tls'), ('h','g'))
            #hilbert space for cavity mode
            self.fock_id = HilbertSpace.register_local_space(make_namespace_string(self.name, 'fock'), range(self.FOCK_DIM))
        except SpaceExists:
            self.tls_id = HilbertSpace.retrieve_by_descriptor(make_namespace_string(self.name, 'tls'))[0]
            self.fock_id = HilbertSpace.retrieve_by_descriptor(make_namespace_string(self.name, 'fock'))[0]
            HilbertSpace.set_states(make_namespace_string(self.name, 'fock'), range(self.FOCK_DIM))
        
        return CavityPort(self) + DecayChannel(self)
        
    def toSLH(self):
        return self.reduce().toSLH()
        

class CavityPort(SubComponent):
    
    def __init__(self, cavity):
        super(CavityPort, self).__init__(cavity, 0)
    
    def toSLH(self):

        sigma_p = LocalSigma(self.tls_id, 'h','g')
        sigma_m = sigma_p.adjoint()

        
        a = Destroy(self.fock_id)
        a_d = a.adjoint()
        
        #coupling to external mode
        L = sqrt(self.kappa) * a
        
        H = self.Delta * sigma_p * sigma_m + 1j * self.g_c * (sigma_p * a - sigma_m * a_d)
        
        return SLH(OperatorMatrixInstance([[1]]), OperatorMatrixInstance([[L]]), H)
        
class DecayChannel(SubComponent):
    
    def __init__(self, cavity):
        super(DecayChannel, self).__init__(cavity, 1)
    
    def toSLH(self):
        
        sigma_p = LocalSigma(self.tls_id, 'h','g')
        sigma_m = sigma_p.adjoint()
        
        # vacuum coupling / spontaneous decay
        L = sqrt(self.gamma) * sigma_m
        
        return SLH(OperatorMatrixInstance([[1]]), OperatorMatrixInstance([[L]]), 0)


def test():
    a = SingleSidedJaynesCummings()
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
