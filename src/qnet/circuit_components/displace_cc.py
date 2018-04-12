"""
Component definition file for a coherent field displacement component.
"""

from sympy.core.symbol import symbols

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import SLH
from qnet.algebra.matrix_algebra import Matrix


__all__ = ['Displace']


class Displace(Component):
    r"""Coherent displacement of the input field (usually vacuum) by a complex
    amplitude :math:`\alpha`.  This component serves as the model of an ideal
    laser source without internal non-classical internal dynamics.
    """

    CDIM = 1

    alpha = symbols('alpha')  # complex valued laser amplitude
    _parameters = ['alpha']

    PORTSIN = ["VacIn"]
    PORTSOUT = ["Out1"]

    def _toSLH(self):

        S = Matrix([[1]])
        L = Matrix([[self.alpha]])
        H = 0

        return SLH(S, L, H)
