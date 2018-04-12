import pytest

from qnet.algebra.state_algebra import BasisKet
from qnet.algebra.hilbert_space_algebra import LocalSpace


def test_operator_times_order():
    psi0_1 = BasisKet(0, hs=1)
    psi1_1 = BasisKet(1, hs=1)
    psi1_2 = BasisKet(1, hs=2)
    psi1_1_m = BasisKet(1, hs=LocalSpace(1, order_index=2))
    psi1_2_m = BasisKet(1, hs=LocalSpace(2, order_index=1))
    bra0_1 = psi0_1.adjoint()
    bra1_1 = psi1_1.adjoint()
    bra1_2 = psi1_2.adjoint()
    bra1_1_m = psi1_1_m.adjoint()
    bra1_2_m = psi1_2_m.adjoint()

    assert psi1_1 + psi0_1 == psi0_1 + psi1_1
    assert (psi1_1 + psi0_1).operands == (psi0_1, psi1_1)
    assert psi1_1 * psi1_2 == psi1_2 * psi1_1
    assert (psi1_1 * psi1_2).operands == (psi1_1, psi1_2)
    assert (psi1_1_m * psi1_2_m).operands == (psi1_2_m, psi1_1_m)
    assert bra1_1 + bra0_1 == bra0_1 + bra1_1
    assert (bra1_1 + bra0_1).ket.operands == (psi0_1, psi1_1)
    assert bra1_1 * bra1_2 == bra1_2 * bra1_1
    assert (bra1_1 * bra1_2).ket.operands == (psi1_1, psi1_2)
    assert (bra1_1_m * bra1_2_m).ket.operands == (psi1_2_m, psi1_1_m)

