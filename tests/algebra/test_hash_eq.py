"""Test hash and equality implementation of Expressions"""

from qnet.algebra.core.circuit_algebra import SLH
from qnet.algebra.library.fock_operators import Destroy
from qnet.algebra.core.hilbert_space_algebra import LocalSpace


def test_equal_hash():
    """Test that expressions with and equal hash or not equal, and that they
    can be used as dictionary keys"""
    a = Destroy(hs="0")
    expr1 = -a
    expr2 = -2 * a
    h1 = hash(expr1)
    h2 = hash(expr2)
    # expr1 and expr2 just happen to have a hash collision for the current
    # implementation of the the hash function. This does not mean that they
    # are not distinguishable!
    assert h1 == h2   # this is the point of the test
    assert expr1 != expr2
    d = {}
    d[expr1] = 1
    d[expr2] = 2
    assert d[expr1] == 1
    assert d[expr2] == 2


def test_heis_eom():
    """Test an example of a Heisenberg EOM that comes out wrong if
    expressions with the same hash are considered equal"""
    import sympy as sp
    a = Destroy(hs="0")
    assert (a-a).is_zero; (a-a) != a-2*a
    heis_eom = SLH([[1]], [sp.sqrt(2)*a], 0).symbolic_heisenberg_eom(a)
    assert heis_eom == -a != -2*a


def test_custom_localspace_identifier_hash():
    """Test hashes for expressions with different local_identifiers for their
    Hilbert spaces have different hashes"""
    hs1 = LocalSpace(1)
    hs1_custom = LocalSpace(1, local_identifiers={'Destroy': 'b'})
    assert hash(hs1) != hash(hs1_custom)
    a = Destroy(hs=hs1)
    b = Destroy(hs=hs1_custom)
    assert hash(a) != hash(b)
