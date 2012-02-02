#!/usr/bin/env python
# encoding: utf-8
"""
mach_zehnder_cc.py

Created automatically
"""
from circuit_components.library import make_namespace_string
from algebra.circuit_algebra import cid, P_sigma, FB
from sympy import symbols
from circuit_components.component import Component


class MachZehnder(Component):
    CDIM = 2
    GENERIC_DEFAULT_VALUES = dict(
        alpha = symbols('alpha', each_char = False),
        phi = symbols('phi', real = True, each_char = False)
                                )
    PORTSIN = ['a', 'b']
    PORTSOUT = ['c', 'd']
    
    arch = "default"
    
    def __init__(self, name = "Q", arch = "default", **params):
        super(MachZehnder, self).__init__(name, **params)
        self.arch = arch
    
    def toSLH(self):
        return self.reduce().toSLH()
        
    def reduce(self):
        return getattr(self, "arch_" + self.arch)()
    
    def arch_netlist(self):
        # import referenced components
        from circuit_components.phase_cc import Phase
        from circuit_components.beamsplitter_cc import Beamsplitter
        from circuit_components.displace_cc import Displace
        
        # instantiate circuit components
        P = Phase(make_namespace_string(self.name, 'P'), phi = self.phi)
        W = Displace(make_namespace_string(self.name, 'W'), alpha = self.alpha)
        B2 = Beamsplitter(make_namespace_string(self.name, 'B2'))
        B1 = Beamsplitter(make_namespace_string(self.name, 'B1'))
        
        return (P_sigma(1, 0) << B2 << (P + cid(1)) << P_sigma(1, 0) << B1 << (W + cid(1)))
    
    
    arch_default = arch_netlist
    
    
if __name__ == "__main__":
    a = MachZehnder()
    print a
    sa = a.toSLH()
    print sa