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

from qnet.algebra.operator_algebra import *
import qutip
import unittest

class TestQutipConversion(unittest.TestCase):

    def testCreateDestoy(self):
        H = local_space("H", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        aq = a.to_qutip()
        for k in range(H.dimension - 1):
            self.assertLess(abs(aq[k, k+1] - sqrt(k + 1)), 1e-10)
        self.assertEqual(ad.to_qutip(),qutip.dag(a.to_qutip()))

    def testN(self):
        H = local_space("H", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        aq = a.to_qutip()
        n = ad * a
        nq = n.to_qutip()
        for k in range(H.dimension):
            self.assertLess(abs(nq[k,k] - k), 1e-10)

    def testSigma(self):
        H = local_space("H", basis = ("e","g","h"))
        sigma = LocalSigma(H, 'g', 'e')
        sq = sigma.to_qutip()
        self.assertEqual(sq[1,0], 1)
        self.assertEqual((sq**2).norm(), 0)

    def testPi(self):
        H = local_space("H", basis = ("e","g","h"))
        Pi_h = LocalProjector(H, 'h')
        self.assertEqual(Pi_h.to_qutip().tr(), 1)
        self.assertEqual(Pi_h.to_qutip()**2, Pi_h.to_qutip())

    def testTensorProduct(self):
        H = local_space("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        H2 = local_space("H2", basis = ("e","g","h"))
        sigma = LocalSigma(H2, 'g', 'e')
        self.assertEqual((sigma * a).to_qutip(), qutip.tensor(a.to_qutip(), sigma.to_qutip()))

    def testLocalSum(self):
        H = local_space("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        self.assertEqual((a + ad).to_qutip(), a.to_qutip() + ad.to_qutip())

    def testNonlocalSum(self):
        H = local_space("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        H2 = local_space("H2", basis = ("e","g","h"))
        sigma = LocalSigma(H2, 'g', 'e')
        self.assertEqual((a + sigma).to_qutip()**2, ((a + sigma)*(a + sigma)).to_qutip())

    def testScalarCoeffs(self):
        H = local_space("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        self.assertEqual(2 * a.to_qutip(), (2 * a).to_qutip())
