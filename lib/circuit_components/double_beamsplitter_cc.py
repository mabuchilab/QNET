#!/usr/bin/env python
# encoding: utf-8
"""
double_beamsplitter_cc.py

Created automatically
"""
from circuit_components.library import make_namespace_string
from algebra.circuit_algebra import cid, P_sigma, FB
from sympy import symbols
from component import Component


class DoubleBeamsplitter(Component):
    
    # total number of field channels
    CDIM = 4
    
    # parameters on which the model depends
    GENERIC_DEFAULT_VALUES = dict(
        theta = 0.7853981633974483
                                )
    # list of input port names
    PORTSIN = ['In1', 'In2', 'In3', 'In4']
    
    # list of output port names
    PORTSOUT = ['Out1', 'Out2', 'Out3', 'Out4']
    
    # architecture to use for reduce(), 
    # only needed, when there are multiple architectures
    arch = "default"
    
    def __init__(self, name = "", arch = "default", **params):
        super(DoubleBeamsplitter, self).__init__(name, **params)
        self.arch = arch
    
    def toSLH(self):
        return self.reduce().toSLH()
        
    def reduce(self):
        return getattr(self, "arch_" + self.arch)()

    # Architectures, i.e. actual implementation of circuit
    
    def arch_netlist(self):
        # import referenced components
        from circuit_components.beamsplitter_cc import Beamsplitter
        
        # instantiate circuit components
        B1 = Beamsplitter(make_namespace_string(self.name, 'B1'), theta = self.theta)
        B2 = Beamsplitter(make_namespace_string(self.name, 'B2'), theta = self.theta)
        
        return ((B2 + (P_sigma(1, 0) << B1)) << P_sigma(3, 2, 0, 1))
    
    
    arch_default = arch_netlist
    
def test():
    a = DoubleBeamsplitter()
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()