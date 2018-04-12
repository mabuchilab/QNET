import pytest

from qnet.algebra.operator_algebra import OperatorSymbol
from qnet.algebra.hilbert_space_algebra import LocalSpace


def test_operator_times_order():
    A1 = OperatorSymbol("A", hs=1)
    B1 = OperatorSymbol("B", hs=1)
    A2 = OperatorSymbol("A", hs=2)
    A3 = OperatorSymbol("A", hs=3)
    B4 = OperatorSymbol("B", hs=4)
    B1_m = OperatorSymbol("B", hs=LocalSpace(1, order_index=2))
    B2_m = OperatorSymbol("B", hs=LocalSpace(2, order_index=1))

    assert A1 * A2 == A2 * A1
    assert A1 * B1 != B1 * A1
    assert (A2 * A1).operands == (A1, A2)
    assert (B2_m * B1_m).operands == (B2_m, B1_m)
    assert ((B4+A3) * (A2+A1)).operands == (A1+A2, A3+B4)

