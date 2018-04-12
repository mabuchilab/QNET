from sympy import symbols, exp, I, pi, cos, sin

from qnet.algebra.circuit_algebra import SLH, Component
from qnet.algebra.matrix_algebra import Matrix

__all__ = ['CoherentDriveCC', 'PhaseCC', 'Beamsplitter']
__private__ = []


class CoherentDriveCC(Component):
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


class PhaseCC(Component):
    r"""Coherent phase shift of the field passing through by real angle
    :math:`\phi`."""

    CDIM = 1

    phi = symbols('phi', real = True)    # PhaseCC angle
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

    def _creduce(self):
        return self


class Beamsplitter(Component):
    r"""Infinite bandwidth beamsplitter model. It is a pure scattering
    component, i.e. it's internal dynamics are not modeled explicitly.  The
    single real parameter theta is the mixing angle for the two signals.  Note
    that there is an asymmetry in the effect on the two input signals due to
    the minus sign appearing in the scattering matrix

    .. math::

        S = \begin{pmatrix} \cos{\theta} & -\sin{\theta} \\ \sin{\theta} & \cos{\theta} \end{pmatrix}

    To achieve a more general beamsplitter combine this component with one or more
    :py:class:`qnet.circuit_components.PhaseCC` components.

    Instantiate as::

        Beamsplitter("B", theta = pi/4)


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
