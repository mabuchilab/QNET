"""Collection of essential circuit components"""
from functools import partial

from sympy import I, cos, exp, pi, sin, symbols

from ..core.circuit_algebra import Component, SLH
from ..core.matrix_algebra import Matrix
from ...utils.properties_for_args import properties_for_args

__all__ = ['CoherentDriveCC', 'PhaseCC', 'Beamsplitter']
__private__ = []


@partial(properties_for_args, arg_names='ARGNAMES')
class CoherentDriveCC(Component):
    r"""Coherent displacement of the input field

    Typically, the input field is the, displaced by a complex
    amplitude :math:`\alpha`. This component serves as the model of an ideal
    laser source without internal non-classical internal dynamics.

    The coherent drive is represented as an inhomogeneous Lindblad operator
    $L = \alpha$, with a trivial Hamiltonian and scattering matrix. For a
    complete circuit with coherent drives, the inhomogeneous Lindblad operators
    can be transformed to driving terms in the total network Hamiltonian
    through :func:`.move_drive_to_H`.

    Args:
        label: label for the component.
        displacement: the coherent displacement amplitude. Defaults to a
            complex symbol 'alpha'
    """

    CDIM = 1  #: circuit dimension
    PORTSIN = ('in', )
    PORTSOUT = ('out', )
    ARGNAMES = ('displacement', )
    DEFAULTS = {'displacement': symbols('alpha')}
    IDENTIFIER = 'W'

    def _toSLH(self):
        S = Matrix([[1]])
        L = Matrix([[self.displacement]])
        return SLH(S, L, 0)


@partial(properties_for_args, arg_names='ARGNAMES')
class PhaseCC(Component):
    r"""Coherent phase shift cicuit component

    The field passing through obtains a phase factor $e^{i \phi}$ for a
    real-valued phase $\phi$. The component has no dynamics, i.e. a trivial
    Hamiltonian and Lindblad operators

    Args:
        label: label for the component.
        phase: the phase. Defaults to a real symbol 'phi'
    """

    CDIM = 1
    PORTSIN = ('in', )
    PORTSOUT = ('out', )
    ARGNAMES = ('phase', )
    DEFAULTS = {'phase': symbols('phi', real=True)}
    IDENTIFIER = 'Phase'

    def _toSLH(self):
        S = Matrix([[exp(I * self.phase)]])
        L = Matrix([[0]])
        return SLH(S, L, 0)


@partial(properties_for_args, arg_names='ARGNAMES')
class Beamsplitter(Component):
    r"""Infinite bandwidth beamsplitter component.

    It is a pure scattering component, i.e. it's internal dynamics are not
    modeled explicitly (trivial Hamiltonian and Lindblad operators). The single
    real parameter is the `mixing_angle` for the two signals.

    .. math::

        S = \begin{pmatrix}
            \cos{\theta} & -\sin{\theta} \\
            \sin{\theta} & \cos{\theta}
            \end{pmatrix}

    The beamsplitter uses the following labeled input/output channels::

             │1:vac
             │
        0:in v 0:tr
        ────>/────>
             │
             │1:rf
             v

    That is, output channel 0 is the transmission of input channel 0 ("in"),
    and output channel 1 is the reflection of input channel 0; vice versa for
    the secondary input channel 1 ("vac": often connected to a vacuum mode).
    For $\theta=0$, the beam splitter results in full transmission, and full
    reflection for $\theta=\pi/2$.

    Args:
        label: label for the beamsplitter.
        mixing_angle: the angle that determines the ratio of transmission and
            reflection defaults to $\pi/4$, corresponding to a
            50-50-beamsplitter. It is recommended to use a sympy expression for
            the mixing angle.

    Note:
        We use a real-valued, but asymmetric scattering matrix. A common
        alternative convention for the beamsplitter is the symmetric scattering
        matrix

        .. math::

            S = \begin{pmatrix}
                \cos{\theta}   & i\sin{\theta} \\
                i \sin{\theta} & \cos{\theta}
                \end{pmatrix}

        To achieve the symmetric beamsplitter (or any general beamsplitter),
        the :class:`Beamsplitter` component can be combined with
        one or more appropriate :class:`PhaseCC` components.
    """
    CDIM = 2  #: circuit dimension
    PORTSIN = ('in', 'vac')
    PORTSOUT = ('tr', 'rf')
    ARGNAMES = ('mixing_angle', )
    DEFAULTS = {'mixing_angle': pi/4}
    IDENTIFIER = 'BS'

    def _toSLH(self):
        theta = self.mixing_angle
        S = Matrix([[cos(theta), -sin(theta)],
                    [sin(theta),  cos(theta)]])
        L = Matrix([[0], [0]])
        return SLH(S, L, 0)
