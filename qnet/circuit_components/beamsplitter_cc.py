#!/usr/bin/env python
# encoding: utf-8
"""
beamsplitter_cc.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""


from qnet.algebra.circuit_algebra import SLH, Matrix, pi, sin, cos, TrivialSpace
from qnet.circuit_components.component import Component


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

    theta = pi/4 # mixing angle, default 50% mixing.
    _parameters = ['theta']

    PORTSIN = ['In1', 'In2']
    PORTSOUT = ['Out1', 'Out2']
    
    def _toSLH(self):
        S = Matrix([[cos(self.theta), -sin(self.theta)],
                    [sin(self.theta),  cos(self.theta)]])
        L = Matrix([[0],
                    [0]])
        return SLH(S, L, 0)

    _space = TrivialSpace

def test():
    a = Beamsplitter("BS")
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    print a.toSLH().tex()
    
if __name__ == "__main__":
    test()