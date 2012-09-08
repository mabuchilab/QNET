#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import Matrix, exp, SLH, I, TrivialSpace
from sympy.core.symbol import symbols





class Phase(Component):
    CDIM = 1

    phi = symbols('phi', real = True)    # Phase angle
    _parameters = ['phi']

    PORTSIN = ["In1"]
    PORTSOUT = ["Out1"]
    
    def _toSLH(self):
        
        S = Matrix([[exp(I * self.phi)]])
        L = Matrix([[0]])
        H = 0
        return SLH(S, L, H)

    def _toABCD(self, linearize):
        return self.toSLH().toABCD(linearize)

    _space = TrivialSpace

    def _creduce(self):
        return self
    

def test():
    a = Phase('P')
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()