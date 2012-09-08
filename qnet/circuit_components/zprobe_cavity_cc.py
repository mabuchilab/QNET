#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

Created by Nikolas Tezak on 2011-02-14.
Copyright (c) 2011 . All rights reserved.
"""

from qnet.algebra.circuit_algebra import (HilbertSpace, local_space, SLH, Matrix,
                                    LocalProjector, Z, sqrt, identity_matrix, LocalSigma)
from qnet.circuit_components.component import Component, SubComponent
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

    gamma = symbols('gamma', real = True)   # decay rate into transverse modes
    Delta = symbols('Delta', real = True)   # detuning between the cavity (mode) and the atomic transition
    _parameters = ['gamma', 'Delta']
            
    PORTSIN = ['PIn', 'FIn1', 'FIn2']
    PORTSOUT = ['POut', 'UOut1', 'UOut2']
    
    sub_blockstructure = (1, 1, 1)
    
    def _creduce(self):
        return ProbePort(self) + FeedbackPort(self, 1) + FeedbackPort(self, 2)

    def _toSLH(self):
        return self.creduce().toSLH()

    @property
    def _space(self):
        return local_space(self.name, self.namespace, basis = ('r','h','g'))



class ProbePort(SubComponent):
    
    def __init__(self, cavity):
        super(ProbePort, self).__init__(cavity, 0)
    
    def _toSLH(self):

        S = Matrix([[Z(self.space)]])
        L = Matrix([[0]])
        H = self.Delta * LocalProjector(self.space, 'r')
        
        return SLH(S, L, H)

class FeedbackPort(SubComponent):
    
    def _toSLH(self):
        
        S =  identity_matrix(1)
        
        if self.sub_index == 1:
            L = sqrt(self.gamma) * Matrix([[LocalSigma(self.space, 'g', 'r')]])
        elif self.sub_index == 2:
            L = sqrt(self.gamma) * Matrix([[LocalSigma(self.space, 'h', 'r')]])
        else:
            raise Exception(str(self.sub_index))

        return SLH(S, L, 0)
        


def test():
    a = ZProbeCavity('Q')
    print a
    print "=" * 80
    print a.creduce()
    print "=" * 80
    print a.toSLH()
    
if __name__ == "__main__":
    test()