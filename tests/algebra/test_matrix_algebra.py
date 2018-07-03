from qnet import (
    Matrix, Zero, One, ZeroOperator, OperatorSymbol, NoConjugateMatrix, zerosm,
    IdentityOperator)
from sympy import symbols, re, im

import pytest


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


def test_matrix_real_imag_conjugate():
    """Test getting a real and imaginary part and conjugate of a matrix"""
    a, b, c, d = symbols('a, b, c, d')

    A = OperatorSymbol('A', hs=0)

    class RealOperatorSymbol(OperatorSymbol):

        def conjugate(self):
            return self

    m = Matrix([[1 + 2j, 1], [-1j, 1j]])
    assert m.real == Matrix([[1, 1], [0, 0]])
    assert m.imag == Matrix([[2, 0], [-1, 1]])
    assert m.conjugate() == Matrix([[1 - 2j, 1], [1j, -1j]])

    m = Matrix([[a, b], [c, d]])
    assert m.real == Matrix([[re(a), re(b)], [re(c), re(d)]])
    assert m.imag == Matrix([[im(a), im(b)], [im(c), im(d)]])
    assert m.conjugate() == Matrix(
        [[a.conjugate(), b.conjugate()],
         [c.conjugate(), d.conjugate()]])

    m = Matrix([[A, b], [c, d]])
    with pytest.raises(NoConjugateMatrix):
        m.real
    with pytest.raises(NoConjugateMatrix):
        m.imag
    with pytest.raises(NoConjugateMatrix):
        m.conjugate()

    A, B, C, D = (RealOperatorSymbol(s, hs=0) for s in ('A', 'B', 'C', 'D'))
    m = Matrix([[A, B], [C, D]])
    assert m.real == m
    assert m.imag == zerosm(m.shape) * IdentityOperator
    assert m.conjugate() == m
