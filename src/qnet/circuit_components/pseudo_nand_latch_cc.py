from qnet.circuit_components.library import make_namespace_string
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import cid, P_sigma, FB
from qnet.circuit_components.pseudo_nand_cc import PseudoNAND


__all__ = ['PseudoNANDLatch']


class PseudoNANDLatch(Component):

    # total number of field channels
    CDIM = 6

    # parameters on which the model depends

    _parameters = []

    # list of input port names
    PORTSIN = ['NS', 'W1', 'kerr2_extra', 'NR', 'W2', 'kerr1_extra']

    # list of output port names
    PORTSOUT = ['BS1_1_out', 'kerr1_out2', 'OUT2_2', 'BS1_2_out', 'kerr2_out2', 'OUT2_1']

    # sub-components

    @property
    def NAND1(self):
        return PseudoNAND(make_namespace_string(self.name, 'NAND1'))

    @property
    def NAND2(self):
        return PseudoNAND(make_namespace_string(self.name, 'NAND2'))

    _sub_components = ['NAND1', 'NAND2']


    def _toSLH(self):
        return self.creduce().toSLH()

    def _creduce(self):

        NAND1, NAND2 = self.NAND1, self.NAND2

        return P_sigma(3, 4, 2, 0, 1, 5) << FB(((P_sigma(0, 1, 3, 2) << NAND2) + cid(3)) << (cid(1) + (P_sigma(0, 3, 4, 5, 1, 2) << ((P_sigma(1, 2, 0, 3) << NAND1) + cid(2)))), out_port=3, in_port=2) << P_sigma(1, 2, 5, 0, 4, 3)

    @property
    def space(self):
        return self.creduce().space
