#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from algebra.circuit_algebra import (HilbertSpace, SpaceExists, SLH, OperatorMatrixInstance, 
                                    LocalProjector, Z, sqrt, IdentityMatrix, LocalSigma)
from circuit_components.component import Component, SubComponent
from sympy.core.symbol import symbols


class ZProbeCavity(Component):
    """
    This is the Z-probe cavity model as used in our group's QEC papers [1,2], which has three (dressed) internal
    states: r, g, h. The qubit is encoded in g,h, while r is used to drive transitions.
    The first channel is the probe-signal, while the second and third channels are 
    the two independent feedback beams. 
    
    For more details see
    [1] http://prl.aps.org/abstract/PRL/v105/i4/e040502
    [2] http://iopscience.iop.org/1367-2630/13/5/055022
    """
    CDIM = 3
    GENERIC_DEFAULT_VALUES = dict(
            gamma = symbols('gamma', real = True, each_char = False), # decay rate into transverse modes
            Delta = symbols('Delta', real = True, each_char = False), # detuning between the cavity (mode) and the atomic transition
            )
            
    PORTSIN = ['PIn', 'FIn1', 'FIn2']
    PORTSOUT = ['POut', 'UOut1', 'UOut2']
    
    sub_blockstructure = (1, 1, 1)
    
    def reduce(self):
        try:
            self.local_space_id = HilbertSpace.register_local_space(self.name, ('r','h','g'))
        except SpaceExists:
            self.local_space_id = HilbertSpace.retrieve_by_descriptor(self.name)[0]
        
        return ProbePort(self) + FeedbackPort(self, 1) + FeedbackPort(self, 2)

    def toSLH(self):
        return self.reduce().toSLH()
    
        

class ProbePort(SubComponent):
    
    def __init__(self, cavity):
        super(ProbePort, self).__init__(cavity, 0)
    
    def toSLH(self):

        S = OperatorMatrixInstance([[Z(self.local_space_id)]])
        L = OperatorMatrixInstance([[0]])
        H = self.Delta * LocalProjector(self.local_space_id, 'r')
        
        return SLH(S, L, H)

class FeedbackPort(SubComponent):
    
    def toSLH(self):
        
        S =  IdentityMatrix(1)
        
        if self.sub_index == 1:
            L = sqrt(self.gamma) * OperatorMatrixInstance([[LocalSigma(self.local_space_id, 'g', 'r')]])
        elif self.sub_index == 2:
            L = sqrt(self.gamma) * OperatorMatrixInstance([[LocalSigma(self.local_space_id, 'h', 'r')]])
        else:
            raise Exception(str(self.sub_index))
        
        return SLH(S, L, 0)
        


def test():
    a = ZProbeCavity('Q')
    print a
    print "=" * 80
    print a.reduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()