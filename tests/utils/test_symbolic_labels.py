"""Test properties of SymbolicLabelBase and its subclasses"""
from qnet import IdxSym, FockIndex, StrLabel, SpinIndex, symbols, SpinSpace
from sympy import IndexedBase


def test_evaluate_symbolic_labels():
    """Test the behavior of the `substitute` method for evaluation of symbolic
    labels"""
    i, j = symbols('i j', cls=IdxSym)
    A = IndexedBase('A')

    lbl = FockIndex(i+j)
    assert lbl.substitute({i: 1, j: 2}) == 3
    assert lbl.substitute({i: 1}) == FockIndex(1+j)
    assert lbl.substitute({j: 2}) == FockIndex(i+2)
    assert lbl.substitute({i: 1}).substitute({j: 2}) == 3
    assert lbl.substitute({}) == lbl

    lbl = StrLabel(A[i, j])
    assert lbl.substitute({i: 1, j: 2}) == 'A_12'
    assert lbl.substitute({i: 1}) == StrLabel(A[1, j])
    assert lbl.substitute({j: 2}) == StrLabel(A[i, 2])
    assert lbl.substitute({i: 1}).substitute({j: 2}) == 'A_12'
    assert lbl.substitute({}) == lbl

    hs = SpinSpace('s', spin=3)
    lbl = SpinIndex(i+j, hs)
    assert lbl.substitute({i: 1, j: 2}) == '+3'
    assert lbl.substitute({i: 1}) == SpinIndex(1+j, hs)
    assert lbl.substitute({j: 2}) == SpinIndex(i+2, hs)
    assert lbl.substitute({i: 1}).substitute({j: 2}) == '+3'
    assert lbl.substitute({}) == lbl

    hs = SpinSpace('s', spin='3/2')
    lbl = SpinIndex((i+j)/2, hs=hs)
    assert lbl.substitute({i: 1, j: 2}) == '+3/2'
    assert lbl.substitute({i: 1}) == SpinIndex((1+j)/2, hs)
    assert lbl.substitute({j: 2}) == SpinIndex((i+2)/2, hs)
    assert lbl.substitute({i: 1}).substitute({j: 2}) == '+3/2'
    assert lbl.substitute({}) == lbl
