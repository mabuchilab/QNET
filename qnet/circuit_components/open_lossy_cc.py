#!/usr/bin/env python
# encoding: utf-8
"""
open_lossy_cc.py

Created automatically
"""
from circuit_components.library import make_namespace_string
from algebra.circuit_algebra import cid, P_sigma, FB
from sympy import symbols
from circuit_components.component import Component


class OpenLossy(Component):
    
    # total number of field channels
    CDIM = 3
    
    # parameters on which the model depends
    GENERIC_DEFAULT_VALUES = dict(
        Delta = symbols('Delta', real = True, each_char = False),
        chi = symbols('chi', real = True, each_char = False),
        kappa = symbols('kappa', real = True, each_char = False),
        theta = symbols('theta', real = True, each_char = False),
        theta_LS0 = symbols('theta_LS0', real = True, each_char = False)
                                )
    # list of input port names
    PORTSIN = ['In1']
    
    # list of output port names
    PORTSOUT = ['Out1', 'Out2']
    
    # architecture to use for creduce(),
    # only needed, when there are multiple architectures
    arch = "default"
    
    def __init__(self, name = "", arch = "default", **params):
        super(OpenLossy, self).__init__(name, **params)
        self.arch = arch
    
    def toSLH(self):
        return self.reduce().toSLH()
        
    def reduce(self):
        return getattr(self, "arch_" + self.arch)()

    # Architectures, i.e. actual implementation of circuit
    
    def arch_netlist(self):
        # import referenced components
        from circuit_components.beamsplitter_cc import Beamsplitter
        from circuit_components.kerr_cavity_cc import KerrCavity
        
        # instantiate circuit components
        KC = KerrCavity(make_namespace_string(self.name, 'KC'), kappa_2 = self.kappa, chi = self.chi, kappa_1 = self.kappa, Delta = self.Delta)
        LSS_ci_ls = Beamsplitter(make_namespace_string(self.name, 'LSS_ci_ls'), theta = self.theta_LS0)
        BS = Beamsplitter(make_namespace_string(self.name, 'BS'), theta = self.theta)
        
        return ((KC + cid(1)) << P_sigma(0, 2, 1) << (LSS_ci_ls + cid(1)) << P_sigma(0, 2, 1) << (BS + cid(1)))
    
    
    arch_default = arch_netlist
    
def test():
    a = OpenLossy()
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()