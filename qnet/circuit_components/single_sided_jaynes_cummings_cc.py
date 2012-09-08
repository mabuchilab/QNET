#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.algebra.circuit_algebra import  Destroy, Matrix, sqrt, SLH, LocalSigma, local_space, I
from qnet.circuit_components.library import make_namespace_string
from qnet.circuit_components.component import Component, SubComponent

from sympy.core.symbol import symbols


class SingleSidedJaynesCummings(Component):
    """
    Typical CQED Jaynes-Cummings model with a single input port with coupling coefficient kappa
    and a single atomic excitation decay 'port' with rate gamma_0.
    """
    
    CDIM = 2
    
    kappa = symbols('kappa', real = True) # decay of cavity mode through cavity mirror
    gamma_0 = symbols('gamma_0', real = True) # decay rate into transverse modes
    g_c = symbols('g_c', real = True)   # coupling between cavity mode and two-level-system
    Delta = symbols('Delta', real = True) # detuning between the cavity (mode) and the atomic transition
    FOCK_DIM = 20
    _parameters = ['kappa', 'gamma_0', 'g_c', 'Delta', 'FOCK_DIM']
    
    
    PORTSIN = ['In1', 'VacIn']
    PORTSOUT = ['Out1', 'UOut']        
    
    sub_blockstructure = (1, 1)


    @property
    def fock_space(self):
        return local_space("f", make_namespace_string(self.namespace, self.name), dimension = self.FOCK_DIM)

    @property
    def tls_space(self):
        return local_space("a", make_namespace_string(self.namespace, self.name), basis = ('h', 'g'))

    @property
    def _space(self):
        return self.fock_space * self.tls_space

    
    def _creduce(self):
        return CavityPort(self) + DecayChannel(self)
        
    def _toSLH(self):
        return self.creduce().toSLH()
        

class CavityPort(SubComponent):
    
    def __init__(self, cavity):
        super(CavityPort, self).__init__(cavity, 0)
    
    def _toSLH(self):

        sigma_p = LocalSigma(self.tls_space, 'h','g')
        sigma_m = sigma_p.adjoint()

        
        a = Destroy(self.fock_space)
        a_d = a.adjoint()
        
        #coupling to external mode
        L = sqrt(self.kappa) * a
        
        H = self.Delta * sigma_p * sigma_m + I * self.g_c * (sigma_p * a - sigma_m * a_d)
        
        return SLH(Matrix([[1]]), Matrix([[L]]), H)
        
class DecayChannel(SubComponent):
    
    def __init__(self, cavity):
        super(DecayChannel, self).__init__(cavity, 1)
    
    def _toSLH(self):
        
        sigma_p = LocalSigma(self.tls_space, 'h','g')
        sigma_m = sigma_p.adjoint()
        
        # vacuum coupling / spontaneous decay
        L = sqrt(self.gamma_0) * sigma_m
        
        return SLH(Matrix([[1]]), Matrix([[L]]), 0)


def test():
    a = SingleSidedJaynesCummings()
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()

    
