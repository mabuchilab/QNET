from qnet.utils.indices import IdxSym
from qnet.algebra.core.abstract_algebra import substitute

import sympy
import pytest


def test_idx_sym():
    """Test basic properties of the IdxSym class"""
    with pytest.raises(TypeError):
        IdxSym(sympy.symbols('i'))
    with pytest.raises(ValueError):
        IdxSym('i', primed=-1)
    with pytest.raises(ValueError):
        IdxSym('1')
    with pytest.raises(ValueError):
        IdxSym('i', primed=-1)
    with pytest.raises(ValueError):
        IdxSym('a^2')
    with pytest.raises(TypeError):
        IdxSym('1', 1)

    assert IdxSym('alpha').name == 'alpha'
    assert IdxSym('alpha_kappa').name == 'alpha_kappa'
    assert IdxSym('alpha_1,2').name == 'alpha_1,2'

    i = IdxSym('i')
    assert i.name == 'i'
    assert i.is_symbol
    assert i.is_integer
    assert i.is_finite
    assert i.is_positive is None
    assert not IdxSym('i', integer=False).is_integer
    assert IdxSym('i', positive=True).is_positive

    assert sympy.sqrt(i**2) == sympy.Abs(i)
    i_pos = IdxSym('i', positive=True)
    assert sympy.sqrt(i_pos**2) == i_pos

    assert i.primed == 0
    assert IdxSym('i', primed=1).primed == 1

    assert i != sympy.symbols('i', integer=True)
    assert i != IdxSym('i', primed=1)
    assert i != IdxSym('i', positive=True)
    assert i != IdxSym('i', integer=False)
    assert i == IdxSym('i', integer=True)
    assert IdxSym('i', primed=1) != IdxSym('i', primed=2)
    assert IdxSym('j', primed=1) != IdxSym('i', primed=1)

    expr = sympy.sqrt(IdxSym('n') + 1)
    expr2 = substitute(expr, {IdxSym('n'): IdxSym('n', primed=1)})
    assert expr2 == sympy.sqrt(IdxSym('n', primed=1) + 1)
