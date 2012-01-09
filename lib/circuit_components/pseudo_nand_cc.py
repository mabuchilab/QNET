#!/usr/bin/env python
# encoding: utf-8
"""
pseudo_nand_cc.py

Created automatically
"""
from circuit_components.library import make_namespace_string
from algebra.circuit_algebra import cid, P_sigma, FB
from sympy import symbols
from component import Component


class PseudoNAND(Component):
    CDIM = 4
    GENERIC_DEFAULT_VALUES = dict(
        phi = symbols('phi', real = True, each_char = False),
        kappa = symbols('kappa', real = True, each_char = False),
        chi = symbols('chi', real = True, each_char = False),
        beta = symbols('beta', each_char = False),
        Delta = symbols('Delta', real = True, each_char = False),
        theta = symbols('theta', real = True, each_char = False)
                                )
    PORTSIN = ['A', 'B', 'VIn1', 'VIn2']
    PORTSOUT = ['UOut1', 'UOut2', 'NAND_AB', 'OUT2']
    
    arch = "default"
    
    def __init__(self, name = "Q", arch = "default", **params):
        super(PseudoNAND, self).__init__(name, **params)
        self.arch = arch
    
    def toSLH(self):
        return self.reduce().toSLH()
        
    def reduce(self):
        return getattr(self, "arch_" + self.arch)()
    
    def arch_netlist(self):
        # import referenced components
        from circuit_components.phase_cc import Phase
        from circuit_components.beamsplitter_cc import Beamsplitter
        from circuit_components.kerr_cavity_cc import KerrCavity
        from circuit_components.displace_cc import Displace
        
        # instantiate circuit components
        P = Phase(make_namespace_string(self.name, 'P'), phi = self.phi)
        K = KerrCavity(make_namespace_string(self.name, 'K'), kappa_2 = self.kappa, chi = self.chi, kappa_1 = self.kappa, Delta = self.Delta)
        W_beta = Displace(make_namespace_string(self.name, 'W_beta'), alpha = self.beta)
        BS2 = Beamsplitter(make_namespace_string(self.name, 'BS2'), theta = self.theta)
        BS1 = Beamsplitter(make_namespace_string(self.name, 'BS1'))
        
        return ((cid(1) + ((cid(1) + ((P + cid(1)) << BS2)) << P_sigma(0, 2, 1) << (K + cid(1)))) << (BS1 + cid(1) + W_beta) << (cid(2) + P_sigma(1, 0)))
    
    
    arch_default = arch_netlist
    
    
if __name__ == "__main__":
    a = PseudoNAND()
    print a
    sa = a.toSLH()
    print sa