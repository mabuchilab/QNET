from qnet.algebra.core.hilbert_space_algebra import LocalSpace
from qnet.algebra.core.operator_algebra import OperatorSymbol, OperatorPlus
from qnet.algebra.toolbox.core import no_instance_caching, temporary_instance_cache


def test_context_instance_caching():
    """Test that we can temporarily suppress instance caching"""
    h1 = LocalSpace("caching")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("a", hs=h1)
    c = OperatorSymbol("c", hs=h1)
    expr1 = a + b
    assert expr1 in OperatorPlus._instances.values()
    with no_instance_caching():
        assert expr1 in OperatorPlus._instances.values()
        expr2 = a + c
        assert expr2 not in OperatorPlus._instances.values()
    with temporary_instance_cache(OperatorPlus):
        assert len(OperatorPlus._instances) == 0
        expr2 = a + c
        assert expr2 in OperatorPlus._instances.values()
    assert expr1 in OperatorPlus._instances.values()
    assert expr2 not in OperatorPlus._instances.values()

