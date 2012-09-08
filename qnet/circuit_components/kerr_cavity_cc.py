#!/usr/bin/env python
# encoding: utf-8
"""
kerr_cavity_cc.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""


from qnet.circuit_components.component import Component, SubComponent
from qnet.circuit_components.library import make_namespace_string
from qnet.algebra.circuit_algebra import HilbertSpace, Destroy, local_space, IdentityMatrix, Matrix, sqrt, SLH, tex, identity_matrix
from sympy.core.symbol import symbols




class KerrCavity(Component):
    """
    Two-port Kerr-nonlinear cavity model with two input and output channels.
    """
    
    CDIM = 2
    
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']
    
    sub_blockstructure = (1, 1)

    name = "K"
    namespace = ""

    Delta = symbols('Delta', real = True)       # Detuning from cavity
    chi = symbols('chi', real = True)           # Kerr-nonlinear coefficient
    kappa_1 = symbols('kappa_1', real = True)   # coupling through first port
    kappa_2 = symbols('kappa_2', real = True)   # coupling through second port
    FOCK_DIM = 75
    _parameters = ['Delta', 'chi', 'kappa_1', 'kappa_2', FOCK_DIM]





    @property
    def _space(self):
        return local_space(self.name, self.namespace, dimension = self.FOCK_DIM)

    @property
    def port1(self):
        return KerrPort(self, 0)

    @property
    def port2(self):
        return KerrPort(self, 1)

    def _creduce(self):
        return self.port1 + self.port2

    def _toSLH(self):
        return self.creduce().toSLH()

    def _toABCD(self, linearize):
        return self.toSLH().toABCD(linearize)
    


class KerrPort(SubComponent):
    
    def _toSLH(self):

        a = Destroy(self.space)
        a_d = a.adjoint()
        S = identity_matrix(1)
        
        if self.sub_index == 0: 
            # Include the Hamiltonian only with the first port of the kerr cavity circuit object
            H = self.Delta * (a_d * a) + self.chi * (a_d * a_d * a * a)
            L = Matrix([[sqrt(self.kappa_1) * a]])
        else:
            H = 0
            L = Matrix([[sqrt(self.kappa_2) * a]])
        
        return SLH(S, L, H)



if __name__ == '__main__':
    a = KerrCavity()
    print a
    sa = a.creduce()
    print "-"*30
    print sa.__repr__()
    print "-"*30
    print sa.toSLH()

    
