from qnet import (
    OperatorSymbol, LocalSpace, IdxSym, symbols, StrLabel, Create, Destroy)


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


def test_indexed_hs_not_disjoint():
    i, j = symbols('i, j', cls=IdxSym)
    hs_i = LocalSpace(StrLabel(i))
    hs_j = LocalSpace(StrLabel(j))
    assert not hs_i.isdisjoint(hs_i)
    assert not hs_i.isdisjoint(hs_j)
    expr = Create(hs=hs_j) * Destroy(hs=hs_i)
    assert expr.args == (Create(hs=hs_j), Destroy(hs=hs_i))
