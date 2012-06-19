#!/usr/bin/env python
# encoding: utf-8
"""
displace.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from circuit_components.component import Component
from algebra.circuit_algebra import *
from sympy.core.symbol import symbols



class Displace(Component):
    """
    Coherent displacement of the input field (usually vacuum).
    This component models an ideal laser source.
    """
    
    CDIM = 1
    
    GENERIC_DEFAULT_VALUES = dict(
        alpha = symbols('alpha', each_char = False) # complex valued laser amplitude
        )
    
    PORTSIN = ["VacIn"]
    PORTSOUT = ["Out1"]
    
    def toSLH(self):
        
        S = OperatorMatrixInstance([[1]])
        L = OperatorMatrixInstance([[self.alpha]])
        H = 0
        
        return SLH(S, L, H)
        
        
    def tex(self):
        return r"{W(%s)}" % tex(self.alpha)
    
#    def mathematica(self):
#        return "CDisplace[%s, Rule[\[Alpha],%s]]" % (self.name, mathematica(self.alpha))

if __name__ == '__main__':
    a = Displace("W")
    print a
    sa = a.reduce()
    print "-"*30
    print sa.__repr__()
    print "-"*30
    print sa.toSLH()


