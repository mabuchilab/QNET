#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""


from circuit_components.component import Component, SubComponent
from algebra.circuit_algebra import HilbertSpace, LocalProjector, SLH, LocalSigma, OperatorMatrixInstance, SpaceExists



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
    
    def reduce(self):
        try:
            self.local_space_id = HilbertSpace.register_local_space(self.name, ('h','g'))
        except SpaceExists:
            self.local_space_id = HilbertSpace.retrieve_by_descriptor(self.name)[0]

        return RelayOut(self) + RelayControl(self)
    
    def toSLH(self):
        return self.reduce().toSLH()



class RelayOut(SubComponent):
    
    def __init__(self, relay):
        super(RelayOut, self).__init__(relay, 0)
        
    def toSLH(self):

        Pi_g = LocalProjector(self.local_space_id, ('g'))
        Pi_h = LocalProjector(self.local_space_id, ('h'))
        
        S = OperatorMatrixInstance([[Pi_g, -Pi_h ], [-Pi_h, Pi_g]])
        return SLH(S, OperatorMatrixInstance([[0]]*2), 0)

class RelayControl(SubComponent):
    
    def __init__(self, relay):
        super(RelayControl, self).__init__(relay, 1)
        
    def toSLH(self):
        
        
        Pi_g = LocalProjector(self.local_space_id, ('g'))
        Pi_h = LocalProjector(self.local_space_id, ('h'))
        sigma_gh = LocalSigma(self.local_space_id, 'g', 'h')
        sigma_hg = LocalSigma(self.local_space_id, 'h', 'g')
        
        
        S = OperatorMatrixInstance([[Pi_g, - sigma_hg ], [-sigma_gh, Pi_h]])
        return SLH(S, OperatorMatrixInstance([[0]]*2), 0)

def test():
    a = Relay('R')
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()