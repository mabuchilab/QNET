from qnet import (
    ScalarTimesOperator, OperatorSymbol, KetSymbol, BraKet, IdxSym,
    KetIndexedSum, BasisKet, FockIndex, LocalSpace, IndexOverRange,
    ZeroOperator, Bra, ZeroKet, Sum, KroneckerDelta)
from qnet.algebra.core.scalar_algebra import (
    Scalar, ScalarExpression, ScalarValue, ScalarPlus, ScalarTimes,
    ScalarPower, sqrt, Zero, One, ScalarIndexedSum)
from sympy import (
    Basic as SympyBasic, symbols, sympify, SympifyError, sqrt as sympy_sqrt,
    IndexedBase, zoo as sympy_infinity, I,
    KroneckerDelta as SympyKroneckerDelta)

import pytest
import numpy as np


@pytest.fixture
def braket():
    """An example symbolic braket"""
    Psi = KetSymbol("Psi", hs=0)
    Phi = KetSymbol("Phi", hs=0)
    res = BraKet.create(Psi, Phi)
    assert isinstance(res, ScalarExpression)
    return res


@pytest.fixture
def a_b_c(braket):
    """Three example scalars for testing algebraic properties"""
    a = braket
    b = ScalarValue(symbols('b'))
    c = ScalarValue(symbols('c'))
    return (a, b, c)


def test_algebraic_properties(a_b_c):
    """Test the basic algebraic properties.

    This is commutativity, associativity, inverse and neutral element of
    summation and multiplicatin, the distributive law, and an involution
    (complex conjugate) with distributivity over sums and products
    """
    # TODO: The same test with a = ScalarValue(symbols('a')) is part of the
    # docs

    a, b, c = a_b_c

    # associativity of summation
    assert a + (b + c) == (a + b) + c
    # commutativity of summation
    assert a * b == b * a
    # neutral element of summation
    assert a + Zero == a + 0 == a
    # inverse of summation
    assert a + (-a) == Zero == 0
    # associativity of product
    assert a * (b * c) == (a * b) * c
    # neutral element of product
    assert a * One == a * 1 == a
    # inverse of product
    assert a * (1/a) == One == 1
    # distributivity
    assert (a * (b + c)).expand() == a * b + a * c
    assert ((a + b) * c).expand() == a * c + b * c
    assert a.conjugate().conjugate() == a
    # distributivity of involution of sum
    assert (a + b).conjugate().expand() == a.conjugate() + b.conjugate()
    # distributivity of involution of product
    assert (a * b).conjugate() == a.conjugate() * b.conjugate()


def test_scalar_numeric_methods(braket):
    """Test all of the numerical magic methods for scalars"""
    three = ScalarValue(3)
    two = ScalarValue(2)
    spOne = sympify(1)
    spZero = sympify(0)
    spHalf = spOne / 2
    assert three == 3
    assert three == three
    assert three != symbols('alpha')
    assert three <= 3
    assert three <= ScalarValue(4)
    assert three >= 3
    assert three >= ScalarValue(2)
    assert three < 3.1
    assert three < ScalarValue(4)
    assert three > ScalarValue(2)
    assert three == sympify(3)
    assert three <= sympify(3)
    assert three >= sympify(3)
    assert three < sympify(3.1)
    assert three > sympify(2.9)
    with pytest.raises(TypeError):
        assert three < symbols('alpha')
    with pytest.raises(TypeError):
        assert three <= symbols('alpha')
    with pytest.raises(TypeError):
        assert three > symbols('alpha')
    with pytest.raises(TypeError):
        assert three >= symbols('alpha')
    assert hash(three) == hash(3)
    v = -three; assert v == -3; assert isinstance(v, ScalarValue)
    v = three + 1; assert v == 4; assert isinstance(v, ScalarValue)
    v = three + two; assert v == 5; assert isinstance(v, ScalarValue)
    v = three + Zero; assert v is three
    assert three + spZero == three
    v = three + One; assert v == 4; assert isinstance(v, ScalarValue)
    assert three + spOne == 4
    v = abs(ScalarValue(-3)); assert v == 3; assert isinstance(v, ScalarValue)
    v = three - 4; assert v == -1; assert isinstance(v, ScalarValue)
    v = three - two; assert v == 1; assert v is One
    v = three - Zero; assert v is three
    assert three - spZero == three
    v = three - One; assert v == 2; assert isinstance(v, ScalarValue)
    assert three - spOne == 2
    v = three * 2; assert v == 6; assert isinstance(v, ScalarValue)
    v = three * two; assert v == 6; assert isinstance(v, ScalarValue)
    v = three * Zero; assert v == 0; assert v is Zero
    assert three * spZero is Zero
    v = three * One; assert v is three
    assert three * spOne == three
    v = three // 2; assert v is One
    assert ScalarValue(3.5) // 1 == 3.0
    v = three // two; assert v is One
    v = three // One; assert v == three
    assert three // spOne == three
    with pytest.raises(ZeroDivisionError):
        v = three // Zero
    with pytest.raises(ZeroDivisionError):
        v = three // spZero
    with pytest.raises(ZeroDivisionError):
        v = three // 0
    v = three / 2; assert v == 3/2; assert isinstance(v, ScalarValue)
    v = three / two; assert v == 3/2; assert isinstance(v, ScalarValue)
    v = three / One; assert v is three
    assert three / spOne == three
    with pytest.raises(ZeroDivisionError):
        v = three / Zero
    with pytest.raises(ZeroDivisionError):
        v = three / spZero
    with pytest.raises(ZeroDivisionError):
        v = three / 0
    v = three % 2; assert v is One
    assert three % 0.2 == 3 % 0.2
    v = three % two; assert v is One
    v = three % One; assert v is Zero
    assert three % spOne is Zero
    with pytest.raises(ZeroDivisionError):
        v = three % Zero
    with pytest.raises(ZeroDivisionError):
        v = three % spZero
    with pytest.raises(ZeroDivisionError):
        v = three % 0
    v = three**2; assert v == 9; assert isinstance(v, ScalarValue)
    v = three**two; assert v == 9; assert isinstance(v, ScalarValue)
    v = three**One; assert v is three
    assert three**spOne == three
    v = three**Zero; assert v is One
    assert three**spZero is One
    v = 1 + three; assert v == 4; assert isinstance(v, ScalarValue)
    v = two + three; assert v == 5; assert isinstance(v, ScalarValue)
    v = sympify(2) + three; assert v == 5; assert isinstance(v, SympyBasic)
    v = 2.0 + three; assert v == 5; assert isinstance(v, ScalarValue)
    v = Zero + three; assert v is three
    with pytest.raises(TypeError):
        None + three
    assert spZero + three == three
    v = One + three; assert v == 4; assert isinstance(v, ScalarValue)
    assert spOne + three == 4
    v = 1 - three; assert v == -2; assert isinstance(v, ScalarValue)
    v = two - three; assert v == -1; assert isinstance(v, ScalarValue)
    v = 2.0 - three; assert v == -1; assert isinstance(v, ScalarValue)
    v = sympify(2) - three; assert v == -1; assert isinstance(v, SympyBasic)
    v = Zero - three; assert v == -3; assert isinstance(v, ScalarValue)
    with pytest.raises(TypeError):
        None - three
    assert spZero - three == -3
    v = One - three; assert v == -2; assert isinstance(v, ScalarValue)
    assert spOne - three == -2
    v = 2 * three; assert v == 6; assert isinstance(v, ScalarValue)
    v = Zero * three; assert v == 0; assert v is Zero
    v = spZero * three; assert v == Zero; assert isinstance(v, SympyBasic)
    v = One * three; assert v is three
    assert spOne * three == three
    with pytest.raises(TypeError):
        None * three
    v = 2 // three; assert v is Zero
    v = two // three; assert v is Zero
    v = One // three; assert v is Zero
    v = spOne // three; assert v == Zero; assert isinstance(v, SympyBasic)
    v = Zero // three; assert v is Zero
    v = spZero // three; assert v == Zero; assert isinstance(v, SympyBasic)
    v = 1 // three; assert v is Zero
    with pytest.raises(TypeError):
        None // three
    v = 2 / three; assert v == 2/3; assert isinstance(v, ScalarValue)
    v = two / three; assert v == 2/3; assert isinstance(v, ScalarValue)
    v = One / three; assert v == 1/3; assert isinstance(v, ScalarValue)
    v = 1 / three; assert v == 1/3; assert isinstance(v, ScalarValue)
    assert spOne / three == 1/3
    v = Zero / three; assert v is Zero
    v = spZero / three; assert v == Zero; assert isinstance(v, SympyBasic)
    with pytest.raises(TypeError):
        None / three
    v = 2**three; assert v == 8; assert isinstance(v, ScalarValue)
    v = 0**three; assert v is Zero
    v = two**three; assert v == 8; assert isinstance(v, ScalarValue)
    v = One**three; assert v is One
    with pytest.raises(TypeError):
        None**three
    v = 1**three; assert v is One
    v = One**spHalf; assert v is One
    v = spOne**three; assert v == One; assert isinstance(v, SympyBasic)
    v = Zero**three; assert v is Zero
    v = spZero**three; assert v == Zero; assert isinstance(v, SympyBasic)
    v = complex(three); assert v == 3+0j; assert isinstance(v, complex)
    v = int(ScalarValue(3.45)); assert v == 3; assert isinstance(v, int)
    v = float(three); assert v == 3.0; assert isinstance(v, float)
    assert Zero == 0
    assert Zero != symbols('alpha')
    assert Zero <= One
    assert Zero <= three
    assert Zero >= Zero
    assert Zero >= -three
    assert Zero < One
    assert Zero < three
    assert Zero > -One
    assert Zero > -three
    assert Zero == spZero
    assert Zero <= spZero
    assert Zero >= spZero
    assert Zero < spOne
    assert Zero > -spOne
    with pytest.raises(TypeError):
        assert Zero < symbols('alpha')
    with pytest.raises(TypeError):
        assert Zero <= symbols('alpha')
    with pytest.raises(TypeError):
        assert Zero > symbols('alpha')
    with pytest.raises(TypeError):
        assert Zero >= symbols('alpha')
    assert hash(Zero) == hash(0)
    assert abs(Zero) is Zero
    assert abs(One) is One
    assert abs(ScalarValue(-1)) is One
    assert -Zero is Zero
    v = -One; assert v == -1; assert isinstance(v, ScalarValue)
    assert Zero + One is One
    assert One + Zero is One
    assert Zero + Zero is Zero
    assert Zero - Zero is Zero
    assert One + One == 2
    assert One - One is Zero
    v = Zero + 2; assert v == 2; assert isinstance(v, ScalarValue)
    v = Zero - One; assert v == -1; assert isinstance(v, ScalarValue)
    v = Zero - 5; assert v == -5; assert isinstance(v, ScalarValue)
    v = 2 + Zero; assert v == 2; assert isinstance(v, ScalarValue)
    v = 2 - Zero; assert v == 2; assert isinstance(v, ScalarValue)
    v = sympify(2) + Zero; assert v == 2; assert isinstance(v, SympyBasic)
    v = sympify(2) - Zero; assert v == 2; assert isinstance(v, SympyBasic)
    v = One + 2; assert v == 3; assert isinstance(v, ScalarValue)
    v = 2 + One; assert v == 3; assert isinstance(v, ScalarValue)
    v = 2 - One; assert v is One
    v = 3 - One; assert v == 2; assert isinstance(v, ScalarValue)
    v = One - 3; assert v == -2; assert isinstance(v, ScalarValue)
    v = sympify(2) + One; assert v == 3; assert isinstance(v, SympyBasic)
    v = sympify(2) - One; assert v == 1; assert isinstance(v, SympyBasic)
    v = sympify(3) - One; assert v == 2; assert isinstance(v, SympyBasic)
    with pytest.raises(TypeError):
        None + Zero
    with pytest.raises(TypeError):
        None - Zero
    with pytest.raises(TypeError):
        None + One
    with pytest.raises(TypeError):
        None - One
    alpha = symbols('alpha')
    assert Zero * alpha is Zero
    v = alpha * Zero; assert v == Zero; assert(isinstance(v, SympyBasic))
    assert 3 * Zero is Zero
    with pytest.raises(TypeError):
        None * Zero
    assert Zero * alpha is Zero
    assert Zero // 3 is Zero
    assert One // 1 is One
    assert One / 1 is One
    assert One == 1
    assert One != symbols('alpha')
    assert One <= One
    assert One <= three
    assert One >= Zero
    assert One >= -three
    assert One < three
    assert One > -three
    assert One == spOne
    assert One <= spOne
    assert One >= spOne
    assert One < sympify(3)
    assert One > -sympify(3)
    with pytest.raises(TypeError):
        assert One < symbols('alpha')
    with pytest.raises(TypeError):
        assert One <= symbols('alpha')
    with pytest.raises(TypeError):
        assert One > symbols('alpha')
    with pytest.raises(TypeError):
        assert One >= symbols('alpha')
    with pytest.raises(ZeroDivisionError):
        One // 0
    with pytest.raises(ZeroDivisionError):
        One / 0
    with pytest.raises(TypeError):
        One // None
    with pytest.raises(TypeError):
        One / None
    with pytest.raises(ZeroDivisionError):
        3 // Zero
    with pytest.raises(TypeError):
        Zero // None
    with pytest.raises(TypeError):
        None // Zero
    assert Zero / 3 is Zero
    with pytest.raises(TypeError):
        Zero / None
    assert Zero % 3 is Zero
    assert Zero % three is Zero
    with pytest.raises(TypeError):
        assert Zero % None
    assert One % 3 is One
    assert One % three is One
    assert three % One is Zero
    assert 3 % One is Zero
    with pytest.raises(TypeError):
        None % 3
    v = sympify(3) % One; assert v == 0; assert isinstance(v, SympyBasic)
    with pytest.raises(TypeError):
        assert One % None
    with pytest.raises(TypeError):
        assert None % One
    assert Zero**2 is Zero
    assert Zero**spHalf is Zero
    with pytest.raises(TypeError):
        Zero**None
    with pytest.raises(ZeroDivisionError):
        v = Zero**-1
    with pytest.raises(ZeroDivisionError):
        v = 1 / Zero
    v = spOne / Zero; assert v == sympy_infinity
    with pytest.raises(ZeroDivisionError):
        v = 1 / Zero
    assert One - Zero is One
    assert Zero * One is Zero
    assert One * Zero is Zero
    with pytest.raises(ZeroDivisionError):
        v = 3 / Zero
    with pytest.raises(ZeroDivisionError):
        v = 3 % Zero
    v = 3 / One; assert v == 3; assert isinstance(v, ScalarValue)
    v = 3 % One; assert v is Zero
    v = 1 % three; assert v is One
    v = spOne % three; assert v == 1; assert isinstance(v, SympyBasic)
    v = sympify(2) % three; assert v == 2
    with pytest.raises(TypeError):
        None % three
    assert 3**Zero is One
    v = 3**One; assert v == 3; assert isinstance(v, ScalarValue)
    v = complex(Zero); assert v == 0j; assert isinstance(v, complex)
    v = int(Zero); assert v == 0; assert isinstance(v, int)
    v = float(Zero); assert v == 0.0; assert isinstance(v, float)
    v = complex(One); assert v == 1j; assert isinstance(v, complex)
    v = int(One); assert v == 1; assert isinstance(v, int)
    v = float(One); assert v == 1.0; assert isinstance(v, float)
    assert braket**Zero is One
    assert braket**0 is One
    assert braket**One is braket
    assert braket**1 is braket
    v = 1 / braket; assert v == braket**(-1); assert isinstance(v, ScalarPower)
    assert v.base == braket
    assert v.exp == -1
    v = three * braket; assert isinstance(v, ScalarTimes)
    assert v == braket * 3
    assert v == braket * sympify(3)
    assert v == 3 * braket
    assert v == sympify(3) * braket
    assert braket * One is braket
    assert braket * Zero is Zero
    assert One * braket is braket
    assert Zero * braket is Zero
    assert spOne * braket is braket
    assert spZero * braket is Zero
    with pytest.raises(TypeError):
        braket // 3
    with pytest.raises(TypeError):
        braket % 3
    with pytest.raises(TypeError):
        1 // braket
    with pytest.raises(TypeError):
        3 % braket
    with pytest.raises(TypeError):
        3**braket
    assert 0**braket is Zero
    assert 1**braket is One
    assert spZero**braket is Zero
    assert spOne**braket is One
    assert One**braket is One
    assert 0 // braket is Zero
    assert 0 / braket is Zero
    assert 0 % braket is Zero
    with pytest.raises(ZeroDivisionError):
        assert 0 / Zero
    with pytest.raises(ZeroDivisionError):
        assert 0 / ScalarValue(0)
    A = OperatorSymbol('A', hs=0)
    v = A / braket; assert isinstance(v, ScalarTimesOperator)
    assert v.coeff == braket**-1
    assert v.term == A
    with pytest.raises(TypeError):
        v = None / braket
    assert braket / three == (1/three) * braket == (spOne/3) * braket
    assert braket / 3 == (1/three) * braket
    v = braket / 0.25; assert v == 4 * braket  # 0.25 and 4 are exact floats
    assert braket / sympify(3) == (1/three) * braket
    assert 3 / braket == 3 * braket**-1
    assert three / braket == 3 * braket**-1
    assert spOne / braket == braket**-1
    braket2 = BraKet.create(KetSymbol("Chi", hs=0), KetSymbol("Psi", hs=0))
    v = braket / braket2; assert v == braket * braket2**-1
    with pytest.raises(ZeroDivisionError):
        braket / Zero
    with pytest.raises(ZeroDivisionError):
        braket / 0
    with pytest.raises(ZeroDivisionError):
        braket / sympify(0)
    assert braket / braket is One
    with pytest.raises(TypeError):
        braket / None
    v = 1 + braket; assert v == braket + 1; assert isinstance(v, Scalar)
    v = One + braket; assert v == braket + One; assert isinstance(v, Scalar)
    assert Zero + braket is braket
    assert spZero + braket is braket
    assert braket + Zero is braket
    assert braket + spZero is braket
    assert 0 + braket is braket
    assert braket + 0 is braket
    assert (-1) * braket == - braket
    assert Zero - braket == - braket
    assert spZero - braket == - braket
    assert braket - Zero is braket
    assert braket - spZero is braket
    assert 0 - braket == - braket
    assert braket - 0 is braket
    assert sympify(3) - braket == 3 - braket


def test_scalar_times_expr_conversion(braket):
    """Test that the coefficient in ScalarTimesQuantumExpression is a Scalar,
    and that Scalar times QuantumExpression is ScalarTimesQantumExpression"""
    # We test with with a ScalarTimesOperator, but this will work for any
    # ScalarTimesQuantumExpression
    A = OperatorSymbol("A", hs=0)
    alpha = symbols('alpha')
    for coeff in (0.5, alpha/2, braket, ScalarValue.create(alpha)):
        for expr in (coeff * A, A * coeff):
            assert isinstance(expr, ScalarTimesOperator)
            assert isinstance(expr.coeff, Scalar)
            assert expr.coeff == coeff
    assert One * A == A
    assert A * One == A
    assert Zero * A is ZeroOperator
    assert A * Zero is ZeroOperator


def test_scalar_plus(braket):
    """Test instantiation of a ScalarPlus expression"""
    expr = 1 + braket
    assert expr == ScalarPlus(ScalarValue(1), braket)
    assert expr.operands == (1, braket)
    assert expr == ScalarPlus.create(braket, ScalarValue(1))
    assert expr == braket + 1

    alpha = symbols('alpha')
    expr = braket - alpha
    assert expr == ScalarPlus(ScalarValue(-alpha), braket)
    assert expr == ScalarPlus.create(braket, ScalarValue(-alpha))

    expr = alpha - braket
    assert expr == ScalarPlus(
        ScalarValue(alpha), ScalarTimes(ScalarValue(-1), braket))
    assert expr == ScalarPlus.create(-braket, alpha)

    expr = braket + braket
    assert expr == ScalarTimes(ScalarValue(2), braket)

    expr = ScalarPlus.create(1, braket, 3)
    assert expr == 4 + braket

    expr = ScalarPlus.create(1, 2, 3)
    assert expr == ScalarValue(6)

    expr = ScalarPlus.create(1, braket, -1)
    assert expr == braket

    expr = ScalarPlus.create(1, braket, -1, 3 * braket)
    assert expr == 4 * braket

    expr = ScalarPlus.create(1, braket, alpha)
    assert expr == (1 + alpha) + braket

    expr = ScalarPlus.create(ScalarValue(1), braket, ScalarValue(alpha))
    assert expr == (1 + alpha) + braket


def test_scalar_times(braket):
    """Test instantiation of a ScalarTimes expression"""
    expr = 2 * braket
    assert expr == ScalarTimes(ScalarValue(2), braket)
    assert expr == ScalarTimes.create(ScalarValue(2), braket)
    assert expr == braket * 2

    expr = ScalarTimes.create(2, braket, 2)
    assert expr == 4 * braket

    half = sympify(1) / 2
    expr = ScalarTimes.create(2, braket, half)
    assert expr == braket

    expr = braket / 2
    assert expr == ScalarTimes(ScalarValue(half), braket)


def test_scalar_power(braket):
    """Test instantiation of a ScalarPower expression"""
    expr = braket * braket
    assert expr == ScalarPower(braket, ScalarValue(2))

    expr = braket**5
    assert expr == ScalarPower(braket, ScalarValue(5))

    expr = (1 + braket)**5
    assert expr == ScalarPower(ScalarPlus(One, braket), ScalarValue(5))

    expr = 2 / braket
    assert expr == ScalarTimes(
        ScalarValue(2), ScalarPower(braket, ScalarValue(-1)))

    assert braket**0 is One
    assert braket**1 == braket


def test_scalar_indexed_sum(braket):
    """Test instantiation and behavior of a ScalarIndexedSum"""
    i = IdxSym('i')
    ip = i.prime
    ipp = ip.prime
    alpha = IndexedBase('alpha')
    a = symbols('a')
    hs = LocalSpace(0)
    ket_sum = KetIndexedSum(
        alpha[1, i] * BasisKet(FockIndex(i), hs=hs),
        IndexOverRange(i, 1, 2))
    bra = KetSymbol('Psi', hs=hs).dag()
    expr = bra * ket_sum
    half = sympify(1) / 2
    assert isinstance(expr, ScalarIndexedSum)
    assert isinstance(expr.term, ScalarTimes)
    assert expr.term == bra * ket_sum.term
    assert expr.ranges == ket_sum.ranges
    assert expr.doit() == (
        alpha[1, 1] * bra * BasisKet(1, hs=hs) +
        alpha[1, 2] * bra * BasisKet(2, hs=hs))

    expr = ScalarIndexedSum.create(i, IndexOverRange(i, 1, 2))
    assert expr == ScalarIndexedSum(i, IndexOverRange(i, 1, 2))
    assert isinstance(expr.doit(), ScalarValue)
    assert expr.doit() == 3

    assert expr.real == expr
    assert expr.imag == Zero
    assert expr.conjugate() == expr

    assert 3 * expr == expr * 3 == Sum(i, 1, 2)(3 * i)
    assert a * expr == expr * a == Sum(i, 1, 2)(a * i)
    assert braket * expr == ScalarTimes(braket, Sum(i, 1, 2)(i))
    assert expr * braket == ScalarTimes(braket, Sum(i, 1, 2)(i))
    assert (2 * i) * expr == 2 * expr * i
    assert (2 * i) * expr == Sum(i, 1, 2)(2 * i * i.prime)

    assert expr * expr == ScalarIndexedSum(
            ScalarValue(i * ip),
            IndexOverRange(i, 1, 2),
            IndexOverRange(ip, 1, 2))

    sum3 = expr**3
    assert sum3 == ScalarIndexedSum(
            ScalarValue(i * ip * ipp),
            IndexOverRange(i, 1, 2),
            IndexOverRange(ip, 1, 2),
            IndexOverRange(ipp, 1, 2))

    assert expr**0 is One
    assert expr**1 is expr
    assert (expr**alpha).exp == alpha
    assert expr**-1 == 1 / expr; assert (1 / expr).exp == -1
    assert (expr**-alpha).exp == -alpha

    sqrt_sum = sqrt(expr)
    assert sqrt_sum == ScalarPower(expr, ScalarValue(half))

    expr = ScalarIndexedSum.create(I * i, IndexOverRange(i, 1, 2))
    assert expr.real == Zero
    assert expr.imag == ScalarIndexedSum.create(i, IndexOverRange(i, 1, 2))
    assert expr.conjugate() == -expr


def test_sqrt(braket):
    """Test QNET's scalar sqrt"""
    half = sympify(1) / 2
    expr = sqrt(braket)
    assert expr == ScalarPower(braket, ScalarValue(half))

    expr = 1 / sqrt(braket)
    assert expr == ScalarPower(braket, ScalarValue(-half))

    braket_abssq = braket * braket.dag()
    expr = sqrt(braket_abssq)
    assert expr**2 == braket_abssq

    assert sqrt(half) == sympy_sqrt(half)
    assert isinstance(sqrt(half), ScalarValue)
    v = sqrt(ScalarValue(half)); assert isinstance(v, ScalarValue)
    assert v == sympy_sqrt(half)
    v = sqrt(2); assert v == sympy_sqrt(2); assert isinstance(v, ScalarValue)
    v = sqrt(0.5); assert v == np.sqrt(0.5); assert isinstance(v, ScalarValue)
    assert sqrt(-1) == sqrt(-One) == sqrt(-sympify(1)) == ScalarValue(I)
    assert isinstance(sqrt(-1), ScalarValue)
    assert sqrt(One) is One
    assert sqrt(sympify(1)) is One
    assert sqrt(Zero) is Zero
    assert sqrt(sympify(0)) is Zero
    with pytest.raises(TypeError):
        assert sqrt(None) is Zero


def test_sympify_scalar(braket):
    """Test that ScalarValue can be converted to sympy"""
    two = ScalarValue.create(2)
    half = sympify(1) / 2
    assert One/2 == half
    alpha = symbols('alpha')
    assert sympify(two) == sympify(2)
    assert sympify(ScalarValue.create(alpha)) == alpha
    with pytest.raises(SympifyError):
        sympify(braket)


def test_zero():
    """Test use of the scalar Zero"""
    alpha = ScalarValue(symbols('alpha'))
    expr = alpha - alpha
    assert expr is Zero
    assert expr == 0
    assert hash(expr) == hash(0)

    expr = alpha + Zero
    assert expr == alpha

    expr = alpha + 0
    assert expr == alpha

    assert ScalarValue.create(0) is Zero
    assert ScalarValue(0) == Zero
    assert sympify(0) == Zero
    assert Zero == sympify(0)
    assert 0 == Zero
    assert Zero == 0
    assert 0j == Zero

    assert Zero.val == 0
    assert len(Zero.args) == 0
    assert Zero.adjoint() == Zero.conjugate() == Zero


def test_scalar_invariant_create(braket):
    """Test that `ScalarValue.create` is invariant w.r.t existing scalars"""
    three = ScalarValue(3)
    assert ScalarValue.create(3) == three == 3
    assert ScalarValue.create(three) is three
    assert ScalarValue.create(braket) is braket
    assert ScalarValue.create(One) is One
    assert ScalarValue.create(Zero) is Zero
    with pytest.raises(TypeError):
        ScalarValue(ScalarValue(3))


def test_values_first(braket):
    """Test that in a product, ScalarValues come before ScalarExpressions"""
    assert ScalarTimes.create(3, braket).operands == (3, braket)
    assert ScalarTimes.create(braket, 3).operands == (3, braket)
    assert isinstance(3 * braket, ScalarTimes)
    assert (3 * braket).operands == (3, braket)
    assert (braket * 3).operands == (3, braket)
    assert (sympify(3) * braket).operands == (3, braket)
    assert (braket * sympify(3)).operands == (3, braket)


def test_one():
    """Test use of the scalar One"""
    alpha = ScalarValue(symbols('alpha'))

    expr = alpha * One
    assert expr == alpha

    expr = alpha * 1
    assert expr == alpha

    expr = alpha / alpha
    assert expr is One
    assert expr == 1
    assert hash(expr) == hash(1)

    assert ScalarValue.create(1) is One
    assert ScalarValue(1) == One
    assert sympify(1) == One
    assert One == sympify(1)
    assert 1 == One
    assert One == 1
    assert 1+0j == One

    assert One.val == 1
    assert len(One.args) == 0
    assert One.adjoint() == One.conjugate() == One


def test_real_complex():
    """Test converting ScalarValue to float/complex"""
    val = ScalarValue(1 - 2j)
    with pytest.raises(TypeError):
        float(val)
    c = complex(val)
    assert c == 1 - 2j
    assert c == val
    assert c.real == 1
    assert c.imag == -2
    assert isinstance(c, complex)

    val = ScalarValue(1.25)
    f = float(val)
    assert f == 1.25
    assert f == val
    assert isinstance(f, float)

    alpha = ScalarValue(symbols('alpha'))
    with pytest.raises(TypeError):
        assert float(alpha) == 0
    with pytest.raises(TypeError):
        assert complex(alpha) == 0


def test_scalar_conjugate(braket):
    """Test taking the complex conjugate (adjoint) of a scalar"""
    Psi = KetSymbol("Psi", hs=0)
    Phi = KetSymbol("Phi", hs=0)
    phi = symbols('phi', real=True)
    alpha = symbols('alpha')

    expr = ScalarValue(1 + 1j)
    assert expr.adjoint() == expr.conjugate() == 1 - 1j

    assert braket.adjoint() == BraKet.create(Phi, Psi)

    expr = 1j + braket
    assert expr.adjoint() == expr.conjugate() == braket.adjoint() - 1j

    expr = (1 + 1j) * braket
    assert expr.adjoint() == expr.conjugate() == (1 - 1j) * braket.adjoint()

    expr = braket**(I * phi)
    assert expr.conjugate() == braket.adjoint()**(-I * phi)

    expr = braket**alpha
    assert expr.conjugate() == braket.adjoint()**(alpha.conjugate())


def test_scalar_real_imag(braket):
    """Test taking the real and imaginary part of a scalar"""
    alpha = symbols('alpha')
    a, b = symbols('a, b', real=True)
    braket_dag = braket.adjoint()

    expr = ScalarValue(1 + 1j)
    assert (expr.real, expr.imag) == (1, 1)

    expr = ScalarValue(a + I * b)
    assert (expr.real, expr.imag) == (a, b)

    expr = ScalarValue(alpha)
    assert (expr.real, expr.imag) == expr.as_real_imag()
    assert (expr.real, expr.imag) == alpha.as_real_imag()

    expr = Zero
    assert (expr.real, expr.imag) == (Zero, Zero)

    expr = One
    assert (expr.real, expr.imag) == (One, Zero)

    assert braket.real == (braket + braket_dag) / 2
    assert braket.imag == (I / 2) * (braket_dag - braket)

    expr = braket + One + I
    assert expr.real.expand().simplify_scalar() == 1 + braket.real.expand()
    assert expr.imag.expand().simplify_scalar() == 1 + braket.imag.expand()

    expr = I * braket
    assert expr.real.expand() == (-I/2) * braket_dag + (I/2) * braket
    assert expr.imag.expand() == braket / 2 + braket_dag / 2

    expr = braket**alpha
    assert expr.real == (expr.adjoint() + expr) / 2
    assert expr.imag == (I/2) * (expr.adjoint() - expr)


def test_differentiation(braket):
    """Test symbolic differentiation of scalars"""
    t = symbols('t', real=True)
    alpha = symbols('alpha')
    expr = ScalarValue(alpha * t**2 / 2 + 2 * t)
    half = sympify(1) / 2
    assert expr.diff(t, 1) == alpha * t + 2
    assert expr.diff(t, 2) == alpha
    assert ScalarValue(2).diff(t, 1) is Zero
    assert ScalarValue(2)._diff(t) is Zero
    assert One.diff(t, 1) is Zero
    assert One._diff(t) is Zero
    assert Zero.diff(t, 1) is Zero
    assert Zero._diff(t) is Zero

    expr = braket * t**2 / 2 + 2 * t
    assert isinstance(expr, Scalar)
    assert expr.diff(t, 1) == braket * t + 2

    expr = sqrt(braket * t)
    assert expr.diff(t, 1) == half * braket * (braket*t)**(-half)
    assert expr.diff(t, 2) == -(half*half) * braket**2 * (braket*t)**(-3*half)

    expr = braket**2
    assert expr.diff(t, 1) is Zero


def test_series_expand(braket):
    """Test expansion of scalar into a series"""
    t = symbols('t', real=True)
    alpha = symbols('alpha')
    three = ScalarValue(3)
    expr = ScalarValue(alpha * t**2 / 2 + 2 * t)

    assert expr.series_expand(t, about=0, order=4) == (
        Zero, 2, alpha/2, Zero, Zero)

    assert expr.series_expand(t, about=0, order=1) == (Zero, 2)

    terms = expr.series_expand(t, about=1, order=4)
    for term in terms:
        assert isinstance(term, Scalar)
    expr_from_terms = (
        sum([terms[i] * (t-1)**i for i in range(1, 5)], terms[0]))
    assert expr_from_terms.val.expand() == expr

    assert expr.series_expand(alpha, about=0, order=4) == (
        2*t, t**2/2, Zero, Zero, Zero)

    assert expr.series_expand(symbols('x'), about=0, order=4) == (
        expr, Zero, Zero, Zero, Zero)

    assert three.series_expand(symbols('x'), 0, 2) == (three, Zero, Zero)
    assert Zero.series_expand(symbols('x'), 0, 2) == (Zero, Zero, Zero)
    assert One.series_expand(symbols('x'), 0, 2) == (One, Zero, Zero)

    expr = One / ScalarValue(t)
    with pytest.raises(ValueError) as exc_info:
        expr.series_expand(t, 0, 2)
    assert "singular" in str(exc_info.value)

    expr = sqrt(ScalarValue(t))
    assert expr.series_expand(t, 1, 2) == (One, One/2, -One/8)
    with pytest.raises(ValueError) as exc_info:
        expr.series_expand(t, 0, 2)
    assert "singular" in str(exc_info.value)

    expr = braket.bra
    assert (
        expr.series_expand(t, 0, 2) ==
        (braket.bra, Bra(ZeroKet), Bra(ZeroKet)))

    expr = braket
    assert expr.series_expand(t, 0, 2) == (braket, Zero, Zero)

    expr = t * braket
    assert expr.series_expand(t, 0, 2) == (Zero, braket, Zero)

    expr = (1 + t * braket)**2
    assert expr.series_expand(t, 0, 2) == (One, 2 * braket, braket**2)

    expr = (1 + t * braket)**(sympify(1)/2)
    with pytest.raises(ValueError):
        expr.series_expand(t, 0, 2)


def test_forwarded_attributes():
    """Test that ScalarValues forward unknown properties/methods to the wrapped
    value"""
    alpha_sym = symbols('alpha', positive=True)
    alpha = ScalarValue(alpha_sym)
    assert alpha.is_positive
    assert alpha.compare(-1) == alpha_sym.compare(-1)
    assert alpha.as_numer_denom() == (alpha_sym, 1)
    with pytest.raises(AttributeError):
        alpha.to_bytes(2, byteorder='big')

    five = ScalarValue(5)
    assert five.to_bytes(2, byteorder='big') == b'\x00\x05'
    with pytest.raises(AttributeError):
        five.is_positive
    with pytest.raises(AttributeError):
        five.as_numer_denom()


def test_kronecker_delta():
    """Test of KroneckerDelta, in addition to the doctest"""
    i, j = IdxSym('i'), IdxSym('j')

    delta_ij = KroneckerDelta(i, j)
    assert isinstance(delta_ij, Scalar)
    assert delta_ij != Zero
    assert delta_ij != One
    assert isinstance(delta_ij.val, SympyKroneckerDelta)
    assert delta_ij.substitute({i: 1, j: 1}) == One
    assert delta_ij.substitute({i: 0, j: 1}) == Zero

    delta_i1 = KroneckerDelta(i, 1)
    assert isinstance(delta_i1, Scalar)
    assert delta_i1 != Zero
    assert delta_i1 != One
    assert isinstance(delta_i1.val, SympyKroneckerDelta)
    assert delta_i1.substitute({i: 1}) == One
    assert delta_i1.substitute({i: 0}) == Zero

    delta_1i = KroneckerDelta(1, i)
    assert isinstance(delta_1i, Scalar)
    assert delta_1i != Zero
    assert delta_1i != One
    assert isinstance(delta_1i.val, SympyKroneckerDelta)
    assert delta_i1.substitute({i: 1}) == One
    assert delta_i1.substitute({i: 0}) == Zero
