#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""


from qnet.circuit_components.component import Component, SubComponent
from qnet.algebra.circuit_algebra import HilbertSpace, LocalProjector, SLH, LocalSigma, Matrix, local_space



GENERIC_DEFAULT_VALUES = {}


class Relay(Component):
    """
    This is the Relay model as used in our group's QEC papers [1,2].
    The SET and RESET inputs control whether the POW input is routed
    through OUT or NOUT.
    
    For more details see
    [1] http://pra.aps.org/abstract/PRA/v80/i4/e045802
    [2] http://prl.aps.org/abstract/PRL/v105/i4/e040502
    """
    
    CDIM = 4
    
    PORTSIN = ['POW', 'VIn', 'SET', 'RESET']
    PORTSOUT = ['NOUT', 'OUT', 'UOut1', 'UOut2']
    
    sub_blockstructure = (2, 2)


    @property
    def _space(self):
        return local_space(self.name, self.namespace, basis = ("h", "g"))
    
    def _creduce(self):
        return RelayOut(self) + RelayControl(self)
    
    def _toSLH(self):
        return self.creduce().toSLH()



class RelayOut(SubComponent):
    
    def __init__(self, relay):
        super(RelayOut, self).__init__(relay, 0)
        
    def _toSLH(self):

        Pi_g = LocalProjector(self.space, 'g')
        Pi_h = LocalProjector(self.space, 'h')
        
        S = Matrix([[Pi_g, -Pi_h ], [-Pi_h, Pi_g]])
        return SLH(S, Matrix([[0]]*2), 0)

class RelayControl(SubComponent):
    
    def __init__(self, relay):
        super(RelayControl, self).__init__(relay, 1)
        
    def _toSLH(self):
        Pi_g = LocalProjector(self.space, 'g')
        Pi_h = LocalProjector(self.space, 'h')

        sigma_gh = LocalSigma(self.space, 'g', 'h')
        sigma_hg = LocalSigma(self.space, 'h', 'g')

        S = Matrix([[Pi_g, - sigma_hg ], [-sigma_gh, Pi_h]])
        return SLH(S, Matrix([[0]]*2), 0)

def test():
    a = Relay('R')
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()