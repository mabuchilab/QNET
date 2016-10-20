#This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QNET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QNET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

from sympy import symbols
import numpy as np
from numpy import sqrt
import qutip
import unittest

from qnet.algebra.operator_algebra import (
        Create, Destroy, LocalSigma, LocalProjector, OperatorSymbol)
from qnet.convert.to_qutip import _time_dependent_to_qutip, convert_to_qutip
from qnet.algebra.hilbert_space_algebra import local_space


class TestQutipConversion(unittest.TestCase):

    def testCreateDestoy(self):
        H = local_space("H", dimension=5)
        ad = Create(H)
        a = Create(H).adjoint()
        aq = convert_to_qutip(a)
        for k in range(H.dimension - 1):
            self.assertLess(abs(aq[k, k+1] - sqrt(k + 1)), 1e-10)
        self.assertEqual(convert_to_qutip(ad), qutip.dag(convert_to_qutip(a)))

    def testN(self):
        H = local_space("H", dimension=5)
        ad = Create(H)
        a = Create(H).adjoint()
        aq = qutip.dag(convert_to_qutip(a))
        self.assertEqual(aq, qutip.create(5))
        n = ad * a
        nq = convert_to_qutip(n)
        for k in range(H.dimension):
            self.assertLess(abs(nq[k,k] - k), 1e-10)

    def testSigma(self):
        H = local_space("H", basis=("e","g","h"))
        sigma = LocalSigma(H, 'g', 'e')
        sq = convert_to_qutip(sigma)
        self.assertEqual(sq[1, 0], 1)
        self.assertEqual((sq**2).norm(), 0)

    def testPi(self):
        H = local_space("H", basis=("e", "g", "h"))
        Pi_h = LocalProjector(H, 'h')
        self.assertEqual(convert_to_qutip(Pi_h).tr(), 1)
        self.assertEqual(convert_to_qutip(Pi_h)**2, convert_to_qutip(Pi_h))

    def testTensorProduct(self):
        H = local_space("H1", dimension=5)
        a = Create(H).adjoint()
        H2 = local_space("H2", basis=("e", "g", "h"))
        sigma = LocalSigma(H2, 'g', 'e')
        self.assertEqual(convert_to_qutip(sigma * a),
                         qutip.tensor(convert_to_qutip(a),
                                      convert_to_qutip(sigma)))

    def testLocalSum(self):
        H = local_space("H1", dimension=5)
        ad = Create(H)
        a = Create(H).adjoint()
        self.assertEqual(convert_to_qutip(a + ad),
                         convert_to_qutip(a) + convert_to_qutip(ad))

    def testNonlocalSum(self):
        H = local_space("H1", dimension=5)
        a = Create(H).adjoint()
        H2 = local_space("H2", basis=("e", "g", "h"))
        sigma = LocalSigma(H2, 'g', 'e')
        self.assertEqual(convert_to_qutip(a + sigma)**2,
                         convert_to_qutip((a + sigma)*(a + sigma)))

    def testScalarCoeffs(self):
        H = local_space("H1", dimension=5)
        a = Create(H).adjoint()
        self.assertEqual(2 * convert_to_qutip(a), convert_to_qutip(2 * a))


    def testSymbol(self):
        expN = OperatorSymbol("expN", 1)
        N = Create(1)*Destroy(1)
        N.space.dimension = 10

        M = Create(2)*Destroy(2)
        M.space.dimension = 5

        converter1 = {
            expN: convert_to_qutip(N).expm()
        }
        expNq = convert_to_qutip(expN, mapping=converter1)

        assert np.linalg.norm(expNq.data.toarray()
            - (convert_to_qutip(N).expm().data.toarray())) < 1e-8

        expNMq = convert_to_qutip(expN*M,  mapping=converter1)

        assert np.linalg.norm(expNMq.data.toarray()
            - (qutip.tensor(convert_to_qutip(N).expm(),
                            convert_to_qutip(M)).data.toarray())) < 1e-8

        converter2 = {
            expN: lambda expr: convert_to_qutip(N).expm()
        }
        expNq = convert_to_qutip(expN, mapping=converter2)

        assert np.linalg.norm(expNq.data.toarray()
            - (convert_to_qutip(N).expm().data.toarray())) < 1e-8

        expNMq = convert_to_qutip(expN*M,  mapping=converter1)

        assert np.linalg.norm(expNMq.data.toarray()
            - (qutip.tensor(convert_to_qutip(N).expm(),
                            convert_to_qutip(M)).data.toarray())) < 1e-8



def test_time_dependent_to_qutip():
    """Test conversion of a time-dependent Hamiltonian"""
    Hil = local_space("H", dimension=5)
    ad = Create(Hil)
    a = Create(Hil).adjoint()

    w, g, t = symbols('w, g, t', real=True)

    H = ad*a + (a + ad)
    assert _time_dependent_to_qutip(H) == convert_to_qutip(H)

    H = g * t * a
    res = _time_dependent_to_qutip(H, time_symbol=t)
    assert res[0] == convert_to_qutip(a)
    assert res[1](1, {}) == g
    assert res[1](1, {g: 2}) == 2

    H =  ad*a + g* t * (a + ad)
    res = _time_dependent_to_qutip(H, time_symbol=t)
    assert len(res) == 3
    assert res[0] == convert_to_qutip(ad*a)
    assert res[1][0] == convert_to_qutip(ad)
    assert res[1][1](1, {}) == g
    assert res[2][0] == convert_to_qutip(a)
    assert res[2][1](1, {}) == g

    res = _time_dependent_to_qutip(H, time_symbol=t, convert_as='str')
    terms = [term for H, term in res[1:]]
    assert terms == ['g*t', 'g*t']

    H =  (ad*a + t * (a + ad))**2
    res = _time_dependent_to_qutip(H, time_symbol=t, convert_as='str')
    assert len(res) == 9
    terms = [term for H, term in res[1:]]
    assert terms == ['t**2', 't', 't', 't**2', '2*t', '2*t**2 + 1', '2*t',
                     't**2']
