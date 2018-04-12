import pytest

from qnet.algebra.hilbert_space_algebra import (
        LocalSpace, TrivialSpace, FullSpace)

def test_product_space_order():
    H1 = LocalSpace(1)
    H2 = LocalSpace('2')
    assert H1 * H2 == H2 * H1
    assert (H1 * H2).operands == (H1, H2)

    H1 = LocalSpace(1)
    H2 = LocalSpace('2', order_index=2)
    H3 = LocalSpace(3, order_index=1)
    assert (H1 * H2 * H3).operands == (H3, H2, H1)
