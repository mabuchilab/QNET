from qnet import Matrix, Zero, One, ZeroOperator, OperatorSymbol
from sympy import symbols


def test_matrixis_zero():
    """Test that zero-matrices can be identified"""
    m = Matrix([[0, 0], [0, 0]])
    assert m.is_zero

    m = Matrix([[Zero, Zero], [Zero, Zero]])
    assert m.is_zero

    m = Matrix([[ZeroOperator, ZeroOperator], [ZeroOperator, ZeroOperator]])
    assert m.is_zero

    m = Matrix([[ZeroOperator, 0], [Zero, ZeroOperator]])
    assert m.is_zero

    m = Matrix([[0, 0], [1, 0]])
    assert not m.is_zero

    m = Matrix([[Zero, Zero], [One, Zero]])
    assert not m.is_zero

    A = OperatorSymbol("A", hs=0)
    m = Matrix([[ZeroOperator, ZeroOperator], [A, ZeroOperator]])
    assert not m.is_zero

    m = Matrix([[ZeroOperator, 0], [symbols('alpha'), ZeroOperator]])
    assert not m.is_zero


def test_matrix_block_structure():
    """Test identification of block structure"""
    m = Matrix([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1]])
    assert m.block_structure == (2, 1, 3)
