#!/usr/bin/env python
# encoding: utf-8
"""
Component definition file for a infinite bandwidth beamsplitter with variable mixing angle.
See :py:class:`Beamsplitter`
"""

import unittest
from qnet.algebra.circuit_algebra import SLH, Matrix, pi, sin, cos, TrivialSpace
from qnet.circuit_components.component import Component




class Beamsplitter(Component):
    r"""
    Infinite bandwidth beamsplitter model. It is a pure scattering component,
    i.e. it's internal dynamics are not modeled explicitly.
    The single real parameter theta is the mixing angle for the two signals.
    Note that there is an asymmetry in the effect on the two input signals due
    to the minus sign appearing in the scattering matrix

    .. math::

        S = \begin{pmatrix} \cos{\theta} & -\sin{\theta} \\ \sin{\theta} & \cos{\theta} \end{pmatrix}

    To achieve a more general beamsplitter combine this component with one or more
    :py:class:`qnet.circuit_components.Phase` components.

    Instantiate as:

    >>> Beamsplitter("B", theta = pi/4)
        Beamsplitter("B", "", pi/4)

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


# Test the circuit
class _TestBeamsplitter(unittest.TestCase):

    def testCreation(self):
        a = Beamsplitter()
        self.assertIsInstance(a, Beamsplitter)

    def testCReduce(self):
        a = Beamsplitter().creduce()

    def testParameters(self):
        if len(Beamsplitter._parameters):
            pname = Beamsplitter._parameters[0]
            obj = Beamsplitter(name="TestName", namespace="TestNamespace", **{pname: 5})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = Beamsplitter(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = Beamsplitter().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()