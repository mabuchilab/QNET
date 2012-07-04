#!/usr/bin/env python
# encoding: utf-8
"""
test_qutip_conversion.py

Created by Nikolas Tezak on 2012-01-07.
Copyright (c) 2012 . All rights reserved.
"""

from qnet.algebra.operator_algebra import *
import qutip
import unittest

class TestQutipConversion(unittest.TestCase):

    def testCreateDestoy(self):
        H = LocalSpace("H", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        aq = a.to_qutip()
        for k in range(H.dimension - 1):
            self.assertLess(abs(aq[k, k+1] - sqrt(k + 1)), 1e-10)
        self.assertEqual(ad.to_qutip(),qutip.dag(a.to_qutip()))

    def testN(self):
        H = LocalSpace("H", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        aq = a.to_qutip()
        n = ad * a
        nq = n.to_qutip()
        for k in range(H.dimension):
            self.assertLess(abs(nq[k,k] - k), 1e-10)

    def testSigma(self):
        H = LocalSpace("H", basis = ("e","g","h"))
        sigma = LocalSigma(H, 'g', 'e')
        sq = sigma.to_qutip()
        self.assertEqual(sq[1,0], 1)
        self.assertEqual((sq**2).norm(), 0)

    def testPi(self):
        H = LocalSpace("H", basis = ("e","g","h"))
        Pi_h = LocalProjector(H, 'h')
        self.assertEqual(Pi_h.to_qutip().tr(), 1)
        self.assertEqual(Pi_h.to_qutip()**2, Pi_h.to_qutip())

    def testTensorProduct(self):
        H = LocalSpace("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        H2 = LocalSpace("H2", basis = ("e","g","h"))
        sigma = LocalSigma(H2, 'g', 'e')
        self.assertEqual((sigma * a).to_qutip(), qutip.tensor(a.to_qutip(), sigma.to_qutip()))

    def testLocalSum(self):
        H = LocalSpace("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        self.assertEqual((a + ad).to_qutip(), a.to_qutip() + ad.to_qutip())

    def testNonlocalSum(self):
        H = LocalSpace("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        H2 = LocalSpace("H2", basis = ("e","g","h"))
        sigma = LocalSigma(H2, 'g', 'e')
        self.assertEqual((a + sigma).to_qutip()**2, ((a + sigma)*(a + sigma)).to_qutip())

    def testScalarCoeffs():
        H = LocalSpace("H1", dimension = 5)
        ad = Create(H)
        a = Create(H).adjoint()
        self.assertEqual(2 * a.to_qutip(), (2 * a).to_qutip())
