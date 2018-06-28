import pytest
import sympy

from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.operator_algebra import (
    OperatorSymbol, tr)
from qnet.algebra.library.fock_operators import Phase, Displace
from qnet.utils.ordering import (
    DisjunctCommutativeHSOrder, FullCommutativeHSOrder, expr_order_key)
from qnet.algebra.core.state_algebra import BraKet, KetBra, BasisKet
from qnet.algebra.core.scalar_algebra import ScalarValue, Zero, One


def test_scalar_expr_order_key():
    """Test that expr_order_key for ScalarValue instances compares just like
    the wrapped values, in particular for Zero and One"""

    half = ScalarValue(0.5)
    two = ScalarValue(2.0)
    alpha = ScalarValue(sympy.symbols('alpha'))
    neg_two = ScalarValue(-2.0)
    neg_alpha = ScalarValue(-sympy.symbols('alpha'))

    key_half = expr_order_key(half)
    key_two = expr_order_key(two)
    key_one = expr_order_key(One)
    key_zero = expr_order_key(Zero)
    key_alpha = expr_order_key(alpha)
    key_neg_two = expr_order_key(neg_two)
    key_neg_alpha = expr_order_key(neg_alpha)

    assert key_half < key_two
    assert key_half < key_one
    assert key_zero < key_half
    assert key_zero < key_one
    assert key_neg_two < key_zero

    # comparison with symbolic should go by string representation, with the
    # nice side-effect that negative symbols are smaller than positive numbers
    assert key_one < key_alpha
    assert key_neg_alpha < key_one
    assert key_zero < key_alpha
    assert key_neg_alpha < key_zero
    assert key_two < key_alpha
    assert key_neg_alpha < key_two
    assert str(-2.0) < "alpha"
    assert key_neg_two < key_alpha
    assert str(-2.0) < "-alpha"
    assert key_neg_two < key_neg_alpha


def disjunct_commutative_test_data():
    A1 = OperatorSymbol("A", hs=1)
    B1 = OperatorSymbol("B", hs=1)
    C1 = OperatorSymbol("C", hs=1)
    A2 = OperatorSymbol("A", hs=2)
    B2 = OperatorSymbol("B", hs=2)
    A3 = OperatorSymbol("A", hs=3)
    B4 = OperatorSymbol("B", hs=4)
    tr_A1 = tr(A1, over_space=1)
    tr_A2 = tr(A2, over_space=2)
    A1_m = OperatorSymbol("A", hs=LocalSpace(1, order_index=2))
    B1_m = OperatorSymbol("B", hs=LocalSpace(1, order_index=2))
    B2_m = OperatorSymbol("B", hs=LocalSpace(2, order_index=1))
    ket_0 = BasisKet(0, hs=1)
    ket_1 = BasisKet(1, hs=1)
    ketbra = KetBra(ket_0, ket_1)
    braket = BraKet(ket_1, ket_1)
    return [
      ([B2, B1, A1],        [B1, A1, B2]),
      ([B2_m, B1_m, A1_m],  [B2_m, B1_m, A1_m]),
      ([B1_m, A1_m, B2_m],  [B2_m, B1_m, A1_m]),
      ([B1, A2, C1, tr_A2], [tr_A2, B1, C1, A2]),
      ([A1, B1+B2],         [A1, B1+B2]),
      ([B1+B2, A1],         [B1+B2, A1]),
      ([A3+B4, A1+A2],      [A1+A2, A3+B4]),
      ([A1+A2, A3+B4],      [A1+A2, A3+B4]),
      ([B4+A3, A2+A1],      [A1+A2, A3+B4]),
      ([tr_A2, tr_A1],      [tr_A1, tr_A2]),
      ([A2, ketbra, A1],    [ketbra, A1, A2]),
      ([A2, braket, A1],    [braket, A1, A2]),
    ]


def full_commutative_test_data():
    A1 = OperatorSymbol("A", hs=1)
    B1 = OperatorSymbol("B", hs=1)
    C1 = OperatorSymbol("C", hs=1)
    A2 = OperatorSymbol("A", hs=2)
    B2 = OperatorSymbol("B", hs=2)
    A3 = OperatorSymbol("A", hs=3)
    B4 = OperatorSymbol("B", hs=3)
    B4 = OperatorSymbol("B", hs=4)
    tr_A1 = tr(A1, over_space=1)
    tr_A2 = tr(A2, over_space=2)
    A1_m = OperatorSymbol("A", hs=LocalSpace(1, order_index=2))
    B1_m = OperatorSymbol("B", hs=LocalSpace(1, order_index=2))
    B2_m = OperatorSymbol("B", hs=LocalSpace(2, order_index=1))
    ket_0 = BasisKet(0, hs=1)
    ket_1 = BasisKet(1, hs=1)
    ketbra = KetBra(ket_0, ket_1)
    braket = BraKet(ket_1, ket_1)
    a = sympy.symbols('a')
    Ph = lambda phi: Phase(phi, hs=1)
    Ph2 = lambda phi: Phase(phi, hs=2)
    D = lambda alpha: Displace(alpha, hs=1)
    return [
      ([B2, B1, A1],             [A1, B1, B2]),
      ([B2_m, B1_m, A1_m],       [B2_m, A1_m, B1_m]),
      ([B1_m, A1_m, B2_m],       [B2_m, A1_m, B1_m]),
      ([B1, A2, C1, tr_A2],      [tr_A2, B1, C1, A2]),
      ([A1, B1+B2],              [A1, B1+B2]),
      ([B1+B2, A1],              [A1, B1+B2]),
      ([A3+B4, A1+A2],           [A1+A2, A3+B4]),
      ([A1+A2, A3+B4],           [A1+A2, A3+B4]),
      ([B4+A3, A2+A1],           [A1+A2, A3+B4]),
      ([tr_A2, tr_A1],           [tr_A1, tr_A2]),
      ([A2, ketbra, A1],         [ketbra, A1, A2]),
      ([A2, braket, A1],         [braket, A1, A2]),
      ([A2, 0.5*A1, 2*A1, A1, a*A1, -3*A1],
                                 [0.5*A1, A1, 2*A1, -3*A1, a*A1, A2]),
      ([Ph(1), Ph(0.5), D(2), D(0.1)],
                                 [D(0.1), D(2), Ph(0.5), Ph(1)]),
      ([Ph(1), Ph2(1), Ph(0.5)], [Ph(0.5), Ph(1), Ph2(1)]),
      ([Ph(a), Ph(1)],           [Ph(1), Ph(a)]),
    ]


@pytest.mark.parametrize('unsorted_args, sorted_args',
                         disjunct_commutative_test_data())
def test_disjunct_commutative_hs_order(unsorted_args, sorted_args):
    res = sorted(unsorted_args,  key=DisjunctCommutativeHSOrder)
    assert res == sorted_args


@pytest.mark.parametrize('unsorted_args, sorted_args',
                         full_commutative_test_data())
def test_full_commutative_hs_order(unsorted_args, sorted_args):
    res = sorted(unsorted_args,  key=FullCommutativeHSOrder)
    assert res == sorted_args
