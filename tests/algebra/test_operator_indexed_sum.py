"""Test indexed sums over operators"""
from sympy import IndexedBase, symbols
from qnet import (
    IdxSym, IndexOverList, OperatorSymbol, OperatorIndexedSum, StrLabel,
    KroneckerDelta)

import pytest


def test_operator_kronecker_sum():
    """Test that Kronecker delta are eliminiated from indexed sums over
    operators"""
    i = IdxSym('i')
    j = IdxSym('j')
    alpha = symbols('alpha')
    delta_ij = KroneckerDelta(i, j)

    def A(i, j):
        return OperatorSymbol(StrLabel(IndexedBase('A')[i, j]), hs=0)

    term = delta_ij * A(i, j)
    sum = OperatorIndexedSum.create(
        term, IndexOverList(i, (1, 2)), IndexOverList(j, (1, 2)))
    assert sum == OperatorIndexedSum.create(A(i, i), IndexOverList(i, (1, 2)))
    assert sum.doit() == (
        OperatorSymbol("A_11", hs=0) + OperatorSymbol("A_22", hs=0))

    term = alpha * delta_ij * A(i, j)
    sum = OperatorIndexedSum.create(
        term, IndexOverList(i, (1, 2)), IndexOverList(j, (1, 2)))
    assert sum == (
        alpha * OperatorIndexedSum.create(A(i, i), IndexOverList(i, (1, 2))))
