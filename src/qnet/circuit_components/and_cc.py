#!/usr/bin/env python
# encoding: utf-8
"""And component"""

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import cid, P_sigma
from qnet.circuit_components.three_port_kerr_cavity_cc import ThreePortKerrCavity
from qnet.circuit_components.beamsplitter_cc import Beamsplitter
from qnet.circuit_components.phase_cc import Phase

__all__ = ['And']


class And(Component):

    # total number of field channels
    CDIM = 4

    # parameters on which the model depends
    Delta = 50.0
    chi = -0.26
    kappa_1 = 20.0
    kappa_2 = 20.0
    kappa_3 = 10.0
    theta = 0.6435
    phi = -1.39
    phip = 2.65
    _parameters = ['Delta', 'chi', 'kappa_1', 'kappa_2', 'kappa_3', 'phi',
                   'phip', 'theta']

    # list of input port names
    PORTSIN = ['In1', 'In2']

    # list of output port names
    PORTSOUT = ['Out1']

    # sub-components

    @property
    def B1(self):
        return Beamsplitter(self.name+'.B1')

    @property
    def B2(self):
        return Beamsplitter(self.name+'.B2', theta=self.theta)

    @property
    def C(self):
        return ThreePortKerrCavity(self.name+'.C', kappa_2=self.kappa_2,
                                   chi=self.chi, kappa_1=self.kappa_1,
                                   kappa_3=self.kappa_3, Delta=self.Delta)

    @property
    def Phase1(self):
        return Phase(self.name+'Phase1', phi=self.phi)

    @property
    def Phase2(self):
        return Phase(self.name+'Phase2', phi=self.phip)

    _sub_components = ['B1', 'B2', 'C', 'Phase1', 'Phase2']

    def _toSLH(self):
        return self.creduce().toSLH()

    def _creduce(self):

        B1, B2, C, Phase1, Phase2 = self.B1, self.B2, self.C, self.Phase1, self.Phase2

        return P_sigma(0, 1, 3, 2) << (((((Phase2 + cid(1)) << P_sigma(1, 0) << B2 << (Phase1 + cid(1))) + cid(1)) << C) + cid(1)) << P_sigma(0, 3, 1, 2) << ((P_sigma(1, 0) << B1) + cid(2))

    @property
    def space(self):
        return self.creduce().space
