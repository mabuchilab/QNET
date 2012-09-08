#!/usr/bin/env python
# encoding: utf-8
"""
displace.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import *
from sympy.core.symbol import symbols



class Displace(Component):
    """
    Coherent displacement of the input field (usually vacuum).
    This component models an ideal laser source.
    """
    
    CDIM = 1
    

    alpha = symbols('alpha') # complex valued laser amplitude
    _parameters = ['alpha']

    
    PORTSIN = ["VacIn"]
    PORTSOUT = ["Out1"]
    
    def _toSLH(self):
        
        S = Matrix([[1]])
        L = Matrix([[self.alpha]])
        H = 0
        
        return SLH(S, L, H)

    _space = TrivialSpace
        
    def _tex(self):
        return r"{W(%s)}" % tex(self.alpha)




if __name__ == '__main__':
    a = Displace("W")
    print a
    sa = a.creduce()
    print "-"*30
    print sa.__repr__()
    print "-"*30
    print sa.toSLH()


