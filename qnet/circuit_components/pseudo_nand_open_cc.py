#!/usr/bin/env python
# encoding: utf-8
"""
pseudo_nand_open_cc.py

Created automatically
"""
from circuit_components.library import make_namespace_string
from algebra.circuit_algebra import cid, P_sigma, FB
from sympy import symbols
from circuit_components.component import Component


class PseudoNANDOpen(Component):
    
    # total number of field channels
    CDIM = 4
    
    # parameters on which the model depends
    GENERIC_DEFAULT_VALUES = dict(
        Delta = symbols('Delta', real = True, each_char = False),
        chi = symbols('chi', real = True, each_char = False),
        kappa = symbols('kappa', real = True, each_char = False),
        phi = symbols('phi', real = True, each_char = False),
        theta = symbols('theta', real = True, each_char = False),
        beta = symbols('beta', each_char = False)
                                )
    # list of input port names
    PORTSIN = ['A', 'B']
    
    # list of output port names
    PORTSOUT = ['NAND_AB', 'OUT2']
    
    # architecture to use for creduce(),
    # only needed, when there are multiple architectures
    arch = "default"
    
    def __init__(self, name = "", arch = "default", **params):
        super(PseudoNANDOpen, self).__init__(name, **params)
        self.arch = arch
    
    def toSLH(self):
        return self.reduce().toSLH()
        
    def reduce(self):
        return getattr(self, "arch_" + self.arch)()

    # Architectures, i.e. actual implementation of circuit
    
    def arch_netlist(self):
        # import referenced components
        from circuit_components.kerr_cavity_cc import KerrCavity
        from circuit_components.phase_cc import Phase
        from circuit_components.beamsplitter_cc import Beamsplitter
        from circuit_components.displace_cc import Displace
        
        # instantiate circuit components
        K = KerrCavity(make_namespace_string(self.name, 'K'), kappa_2 = self.kappa, chi = self.chi, kappa_1 = self.kappa, Delta = self.Delta)
        P = Phase(make_namespace_string(self.name, 'P'), phi = self.phi)
        W_beta = Displace(make_namespace_string(self.name, 'W_beta'), alpha = self.beta)
        BS1 = Beamsplitter(make_namespace_string(self.name, 'BS1'))
        BS2 = Beamsplitter(make_namespace_string(self.name, 'BS2'), theta = self.theta)
        
        return ((cid(2) + P_sigma(1, 0)) << (((((P + cid(1)) << BS2 << (W_beta + cid(1))) + cid(1)) << (cid(1) + (P_sigma(1, 0) << K))) + cid(1)) << (cid(1) + (P_sigma(0, 2, 1) << ((P_sigma(1, 0) << BS1) + cid(1)))) << P_sigma(0, 3, 2, 1))
    
    
    arch_default = arch_netlist
    
def test():
    a = PseudoNANDOpen()
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()