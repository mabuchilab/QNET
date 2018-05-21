from qnet import (
    One, Zero, ZeroOperator, IdentityOperator, ZeroSuperOperator,
    IdentitySuperOperator, ZeroKet, TrivialKet, FullSpace, TrivialSpace,
    CIdentity, CircuitZero)


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
