from qnet import (
    One, Zero, ZeroOperator, IdentityOperator, ZeroSuperOperator,
    IdentitySuperOperator, ZeroKet, TrivialKet, FullSpace, TrivialSpace,
    CIdentity, CircuitZero, IdxSym, BasisKet, OperatorSymbol, FockIndex,
    KetIndexedSum, OperatorIndexedSum, StrLabel, LocalSpace,
    IndexOverList, IndexOverFockSpace, IndexOverRange, Sum)
from sympy import IndexedBase


def test_neutral_elements():
    """test the properties of the neutral elements in the quantum algebras.
    This tests the resolution of #63

    *Only* the scalar neutral elements compare to and hash as 0 and 1. The
    neutral elements of all other algebras are "unique" and don't compare to 0
    and 1. Elements of a quantum algebra have an is_zero attribute
    """
    assert One == 1
    assert One is not 1
    assert hash(One) == hash(1)
    assert Zero == 0
    assert Zero is not 0
    assert hash(Zero) == hash(0)
    assert Zero.is_zero

    assert IdentityOperator != 1
    assert hash(IdentityOperator) != hash(1)
    assert ZeroOperator != 0
    assert hash(ZeroOperator) != hash(0)
    assert ZeroOperator.is_zero

    assert IdentitySuperOperator != 1
    assert hash(IdentitySuperOperator) != hash(1)
    assert ZeroSuperOperator != 0
    assert hash(ZeroSuperOperator) != hash(0)
    assert ZeroSuperOperator.is_zero

    assert TrivialKet != 1
    assert hash(TrivialKet) != hash(1)
    assert ZeroKet != 0
    assert hash(ZeroKet) != hash(0)
    assert ZeroKet.is_zero

    #  the remainder are not quantum algebra elements, to they don't have
    #  is_zero
    assert FullSpace != 1
    assert hash(FullSpace) != hash(1)
    assert TrivialSpace != 0
    assert hash(TrivialSpace) != hash(0)

    assert CIdentity != 1
    assert hash(CIdentity) != hash(1)
    assert CircuitZero != 0
    assert hash(CircuitZero) != hash(0)


def test_sum_instantiator():
    """Test use of Sum instantiator"""
    i = IdxSym('i')
    j = IdxSym('j')
    ket_i = BasisKet(FockIndex(i), hs=0)
    ket_j = BasisKet(FockIndex(j), hs=0)
    A_i = OperatorSymbol(StrLabel(IndexedBase('A')[i]), hs=0)
    hs0 = LocalSpace('0')

    sum = Sum(i)(ket_i)
    ful = KetIndexedSum(ket_i, IndexOverFockSpace(i, hs=hs0))
    assert sum == ful
    assert sum == Sum(i, hs0)(ket_i)
    assert sum == Sum(i, hs=hs0)(ket_i)

    sum = Sum(i, 1, 10)(ket_i)
    ful = KetIndexedSum(ket_i, IndexOverRange(i, 1, 10))
    assert sum == ful
    assert sum == Sum(i, 1, 10, 1)(ket_i)
    assert sum == Sum(i, 1, to=10, step=1)(ket_i)
    assert sum == Sum(i, 1, 10, step=1)(ket_i)

    sum = Sum(i, (1, 2, 3))(ket_i)
    ful = KetIndexedSum(ket_i, IndexOverList(i, (1, 2, 3)))
    assert sum == KetIndexedSum(ket_i, IndexOverList(i, (1, 2, 3)))
    assert sum == Sum(i, [1, 2, 3])(ket_i)

    sum = Sum(i)(Sum(j)(ket_i * ket_j.dag()))
    ful = OperatorIndexedSum(
        ket_i * ket_j.dag(),
        IndexOverFockSpace(i, hs0), IndexOverFockSpace(j, hs0))
    assert sum == ful

    #sum = Sum(i)(Sum(j)(ket_i.dag() * ket_j)) # TODO
    #assert sum == ful

    # TODO: sum over A_i
