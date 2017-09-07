# This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QNET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QNET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################

import pytest

from sympy import symbols, sqrt, exp, I, Rational
from sympy.core import S

from qnet.printing.sympy import SympyUnicodePrinter, derationalize_denom


def test_derationalize_denom():
    assert derationalize_denom(1/sqrt(2)) == (1, 2, S.One)
    assert derationalize_denom(sqrt(2)/2) == (1, 2, S.One)
    assert derationalize_denom(3/sqrt(2)) == (3, 2, S.One)
    assert (derationalize_denom(exp(I*symbols('phi'))/sqrt(2)) ==
            (1, 2, exp(I*symbols('phi'))))
    with pytest.raises(ValueError) as exc_info:
        derationalize_denom(1 + 3/sqrt(2))
    assert str(exc_info.value) == 'expr is not a Mul instance'
    with pytest.raises(ValueError) as exc_info:
        derationalize_denom(Rational(5, 4) * sqrt(3))
    assert str(exc_info.value) == 'Cannot derationalize'
    with pytest.raises(ValueError):
        derationalize_denom(sqrt(3)/sqrt(2))



@pytest.mark.parametrize("expr,expected_str", [
    (symbols('alpha'),
        'α'),
    (symbols("alpha_1"),
        'α₁'),
    (symbols("alpha_1")**2,
        'α₁²'),
    (symbols("alpha_1")**(symbols('n') + 1),
        'α₁⁽ⁿ ⁺ ¹⁾'),
    (symbols("alpha_1")**((symbols('n') + 1) / 2),
        'α₁**(n/2 + 1/2)'),
    (I * symbols("alpha_1"),
        'ⅈ α₁'),
    (sqrt(2),
        '√2'),
    (3/sqrt(2),
        '3/√2'),
    (symbols("alpha_1") / sqrt(2),
        'α₁ / √2'),
    (I * symbols("alpha_1") / sqrt(2),
        '(ⅈ α₁) / √2'),
    ((symbols('x') + I * symbols('y')) / sqrt(2),
        '(x + ⅈ y) / √2'),
    (sqrt(symbols('mu_1') - symbols('mu_2')) * symbols("alpha_1"),
        'α₁ √(μ₁ - μ₂)'),
    (exp(-I * symbols("phi")),
        'exp(-ⅈ φ)'),
    (exp(-I * symbols("phi")) / (1 - symbols('eta_0'))**2,
        'exp(-ⅈ φ)/(-η₀ + 1)²'),
])
def test_sympy_unicode(expr, expected_str):
    #stop = I * symbols("alpha_1") / sqrt(2) # DEBUG
    #if expr == stop: # DEBUG
        #from IPython.terminal.debugger import set_trace; set_trace() # DEBUG
    out_str = SympyUnicodePrinter().doprint(expr)
    assert out_str == expected_str
