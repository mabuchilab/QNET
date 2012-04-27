#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from circuit_components.component import Component
from algebra.circuit_algebra import OperatorMatrixInstance, exp, SLH
from sympy.core.symbol import symbols
from sympy import I
from algebra.abstract_algebra import mathematica



class Phase(Component):
    CDIM = 1
    GENERIC_DEFAULT_VALUES = dict(
        phi = symbols('phi', each_char = False, real = True)    # Phase angle
        )
    PORTSIN = ["In1"]
    PORTSOUT = ["Out1"]
    
    def toSLH(self):
        
        S = OperatorMatrixInstance([[exp(I * self.phi)]])
        L = OperatorMatrixInstance([[0]])
        H = 0
        
        return SLH(S, L, H)
    
#    def mathematica(self):
#        return "CPhase[%s, Rule[\[Phi],%s]]" % (self.name, mathematica(self.phi))

def test():
    a = Phase('P')
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()