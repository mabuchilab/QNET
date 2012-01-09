#!/usr/bin/env python
# encoding: utf-8
"""
test_qutip_conversion.py

Created by Nikolas Tezak on 2012-01-07.
Copyright (c) 2012 . All rights reserved.
"""

from algebra.operator_algebra import Create, LocalProjector, LocalSigma, HilbertSpace, SpaceExists
import qutip
from math import sqrt

try:
    H2id = HilbertSpace.register_local_space('H2', ('e','g','h'))
    H2 = HilbertSpace((H2id,))
except SpaceExists:
    H2 = HilbertSpace(("H2",))

try:
    Hid = HilbertSpace.register_local_space('H', range(5))
    H = HilbertSpace((Hid,))
except SpaceExists:
    H = HilbertSpace(("H",))


ad = Create(H)
a = Create(H).adjoint()


def test_destroy():
    aq = a.to_qutip()
    for k in range(H.dimension - 1):
        assert abs(aq[k, k+1] - sqrt(k + 1)) < 1e-10


def test_create():
    assert ad.to_qutip() == qutip.dag(a.to_qutip())
    

n = ad * a

def test_n():
    nq = n.to_qutip()
    for k in range(H.dimension):
        assert abs(nq[k,k] - k) < 1e-10


sigma = LocalSigma(H2, 'g', 'e')

def test_sigma():
    sq = sigma.to_qutip()
    assert sq[1,0] == 1
    assert (sq**2).norm() == 0



Pi_h = LocalProjector(H2, 'h')
def test_Pi():
    assert Pi_h.to_qutip().tr() == 1
    assert Pi_h.to_qutip()**2 == Pi_h.to_qutip()


def test_tensor_product():    
    assert (sigma * a).to_qutip() == qutip.tensor(sigma.to_qutip(), a.to_qutip())




def test_local_sum():
    assert (a + ad).to_qutip() == a.to_qutip() + ad.to_qutip()



def test_nonlocal_sum():
    assert (a + sigma).to_qutip()**2 == ((a + sigma)*(a + sigma)).to_qutip()




def test_scalar_coeffs():
    assert 2 * a.to_qutip() == (2 * a).to_qutip()

if __name__ == '__main__':
    keys = globals().keys()
    for fn in keys:
        if fn.startswith('test_'):
            globals()[fn]()