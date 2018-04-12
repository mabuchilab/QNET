"""Component definition file for a pseudo-delay model that works over a limited
bandwidth.  See documentation of :py:class:`Delay`.
"""

from functools import reduce as freduce

from sympy.core.symbol import symbols
from sympy import sqrt
from numpy import array as np_array

from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import SLH
from qnet.algebra.matrix_algebra import Matrix
from qnet.algebra.operator_algebra import Create, Destroy, ZeroOperator


__all__ = ['Delay']


class Delay(Component):
    r"""Delay"""

    CDIM = 1

    tau = symbols('tau', positive = True) # positive valued delay
    N = 3
    FOCK_DIM = 25

    _parameters = ['alpha', 'N', 'FOCK_DIM']

    PORTSIN = ["In1"]
    PORTSOUT = ["Out1"]

    def _toSLH(self):

        # These numerically optimal solutions were obtained as outlined in
        # my blog post on the Mabuchi-Lab internal blog
        # email me (ntezak@stanford.edu)for details.
        if self.N == 1:
            kappa0 = 9.28874141848 / self.tau
            kappas = np_array([7.35562929]) / self.tau
            Deltas = np_array([3.50876192]) / self.tau
        elif self.N == 3:
            kappa0 = 14.5869543803 / self.tau
            kappas = np_array([ 13.40782559, 9.29869721]) / self.tau
            Deltas = np_array([3.48532283, 7.14204585]) / self.tau
        elif self.N == 5:
            kappa0 = 19.8871474779 / self.tau
            kappas = np_array([19.03316217, 10.74270752, 16.28055664]) / self.tau
            Deltas = np_array([3.47857213, 10.84138821, 7.03434809]) / self.tau
        else:
            raise NotImplementedError("The number of cavities to realize the delay must be one of 1,3 or 5.")

        h0 = self.name+'C0'
        hp =  [self.name+".C{:d}p".format(n+1) for n in range((self.N-1)//2)]
        hm =  [self.name+".C{:d}m".format(n+1) for n in range((self.N-1)//2)]

        S = Matrix([1.])
        slh0 = SLH(S, Matrix([[sqrt(kappa0) * Destroy(hs=h0)]]), ZeroOperator)
        slhp = [SLH(S, Matrix([[sqrt(kj) * Destroy(hs=hj)]]), Dj * Create(hs=hj) * Destroy(hs=hj)) for (kj, Dj, hj) in zip(kappas, Deltas, hp)]
        slhm = [SLH(S, Matrix([[sqrt(kj) * Destroy(hs=hj)]]), -Dj * Create(hs=hj) * Destroy(hs=hj)) for (kj, Dj, hj) in zip(kappas, Deltas, hm)]

        return freduce(lambda a, b: a << b, slhp + slhm, slh0).toSLH()
