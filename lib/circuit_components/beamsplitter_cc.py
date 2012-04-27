#!/usr/bin/env python
# encoding: utf-8
"""
beamsplitter_cc.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from algebra.circuit_algebra import SLH, OperatorMatrixInstance, pi, sin, cos, mathematica
from circuit_components.component import Component
from collections import OrderedDict

class Beamsplitter(Component):
    """
    Beamsplitter model. 
    The single real parameter theta is the mixing angle for the two signals.
    Note that there is an asymmetry in the effect on the two input signals due
    to the minus sign appearing in the scattering matrix S.
    To achieve a more general beamsplitter as in [1] combine this component with one or more
    Phase components.
    
    [1] http://dx.doi.org/10.1109/TAC.2009.2031205
    """
    CDIM = 2
    GENERIC_DEFAULT_VALUES = OrderedDict(
                                theta = pi/4, # mixing angle, default 50% mixing.
                                )
    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']
    
    def toSLH(self):
        S = OperatorMatrixInstance([[cos(self.theta), -sin(self.theta)], 
                                    [sin(self.theta),  cos(self.theta)]])
        L = OperatorMatrixInstance([[0],
                                    [0]])
        return SLH(S, L, 0)
    
    def mathematica(self):
        return r"Beamsplitter[%s, Rule[\[Theta],%s]]" % (self.name, mathematica(self.theta))
        

def test():
    a = Beamsplitter()
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    print a.tex()
    
if __name__ == "__main__":
    test()