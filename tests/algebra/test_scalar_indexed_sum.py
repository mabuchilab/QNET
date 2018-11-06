from sympy import IndexedBase
from qnet import (
    symbols, IdxSym, KroneckerDelta, ScalarValue, LocalSpace,
    IndexOverFockSpace, ScalarIndexedSum, ScalarTimes, KetSymbol, StrLabel,
    BraKet)
from qnet.algebra.core.algebraic_properties import indexed_sum_over_kronecker


def test_indexed_sum_over_kronecker_scalarvalue():
    """Test indexed_sum_over_kronecker for a ScalarValue.

    This is an auxiliary test to resolve a bug in test_tls_norm
    """
    i = IdxSym('i')
    ip = i.prime
    term = KroneckerDelta(i, ip) / 2
    assert isinstance(term, ScalarValue)
    hs = LocalSpace('tls', dimension=2)
    i_range = IndexOverFockSpace(i, hs)
    ip_range = IndexOverFockSpace(ip, hs)
    sum = indexed_sum_over_kronecker(
        ScalarIndexedSum, (term, i_range, ip_range), {})
    assert sum == 1


def test_indexed_sum_over_scalartimes():
    """Test ScalarIndexedSum over a term that is an ScalarTimes instance"""
    i, j = symbols('i, j', cls=IdxSym)
    hs = LocalSpace(1, dimension=2)
    Psi_i = KetSymbol(StrLabel(IndexedBase('Psi')[i]), hs=hs)
    Psi_j = KetSymbol(StrLabel(IndexedBase('Psi')[j]), hs=hs)
    term = KroneckerDelta(i, j) * BraKet(Psi_i, Psi_j)
    assert isinstance(term, ScalarTimes)
    i_range = IndexOverFockSpace(i, hs)
    j_range = IndexOverFockSpace(j, hs)
    sum = ScalarIndexedSum.create(term, i_range, j_range)
    assert sum == hs.dimension
