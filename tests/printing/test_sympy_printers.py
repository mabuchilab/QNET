import pytest

import sympy
from sympy import symbols, sqrt, exp, I, Rational, IndexedBase
from sympy.core import S
from qnet.utils.indices import IdxSym

import qnet
from qnet.printing.sympy import (
    SympyUnicodePrinter, SympyLatexPrinter, SympyStrPrinter,
    derationalize_denom)


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


def test_primed_IdxSym():
    """Test that primed IdxSym are rendered correctly not just in QNET's
    printing system, but also in SymPy's printing system"""
    ipp = IdxSym('i').prime.prime
    assert qnet.ascii(ipp) == "i''"
    assert qnet.latex(ipp) == r'{i^{\prime\prime}}'
    assert qnet.srepr(ipp) == "IdxSym('i', integer=True, primed=2)"
    assert qnet.unicode(ipp) == "i''"
    assert sympy.printing.sstr(ipp) == qnet.ascii(ipp)
    assert sympy.printing.latex(ipp) == qnet.latex(ipp)
    assert sympy.printing.srepr(ipp) == qnet.srepr(ipp)
    assert sympy.printing.pretty(ipp) == qnet.unicode(ipp)


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
    (IndexedBase('alpha')[IdxSym('i'), IdxSym('j')],
        'α_ij'),
    (IndexedBase('alpha')[IdxSym('mu'), IdxSym('nu')],
        'α_μν'),
    (IndexedBase('alpha')[1, 10],
        'α_1,10'),
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
    (IdxSym('i'),
        'i'),
    (IdxSym('alpha'),
        'α'),
    (IdxSym('alpha_1'),
        'α₁'),
    (IdxSym('alpha', primed=2),
        "α''"),
    (sqrt(IdxSym('n')+1),
        '√(n + 1)'),
    (sqrt(IdxSym('n', primed=1)+1),
        "√(n' + 1)"),
    (IdxSym('n', primed=1)**2,
        "n'²"),
])
def test_sympy_unicode(expr, expected_str):
    out_str = SympyUnicodePrinter().doprint(expr)
    assert out_str == expected_str


def test_unicode_parenthization():
    """Test that parenthesize does not return a prettyForm"""
    alpha = symbols('alpha')
    printer = SympyUnicodePrinter()
    printer.parenthesize(alpha, 0) == 'α'


@pytest.mark.parametrize("expr,expected_str", [
    (symbols('alpha'),
        'alpha'),
    (symbols("alpha_1"),
        'alpha_1'),
    (symbols("alpha_1")**2,
        'alpha_1**2'),
    (symbols("alpha_1")**(symbols('n') + 1),
        'alpha_1**(n + 1)'),
    (symbols("alpha_1")**((symbols('n') + 1) / 2),
        'alpha_1**(n/2 + 1/2)'),
    (IndexedBase('alpha')[IdxSym('i'), IdxSym('j')],
        'alpha_ij'),
    (IndexedBase('alpha')[IdxSym('mu'), IdxSym('nu')],
        'alpha_mu,nu'),
    (IndexedBase('alpha')[1, 10],
        'alpha_1,10'),
    (I * symbols("alpha_1"),
        'I*alpha_1'),
    (sqrt(2),
        'sqrt(2)'),
    (3/sqrt(2),
        '3/sqrt(2)'),
    (symbols("alpha_1") / sqrt(2),
        'alpha_1 / sqrt(2)'),
    (I * symbols("alpha_1") / sqrt(2),
        '(I*alpha_1) / sqrt(2)'),
    ((symbols('x') + I * symbols('y')) / sqrt(2),
        '(x + I*y) / sqrt(2)'),
    (sqrt(symbols('mu_1') - symbols('mu_2')) * symbols("alpha_1"),
        'alpha_1*sqrt(mu_1 - mu_2)'),
    (exp(-I * symbols("phi")),
        'exp(-I*phi)'),
    (exp(-I * symbols("phi")) / (1 - symbols('eta_0'))**2,
        'exp(-I*phi)/(-eta_0 + 1)**2'),
    (IdxSym('i'),
        'i'),
    (IdxSym('alpha'),
        'alpha'),
    (IdxSym('alpha_1'),
        'alpha_1'),
    (IdxSym('alpha', primed=2),
        "alpha''"),
    (sqrt(IdxSym('n')+1),
        'sqrt(n + 1)'),
    (sqrt(IdxSym('n', primed=1)+1),
        "sqrt(n' + 1)"),
    (IdxSym('n', primed=1)**2,
        "n'**2"),
])
def test_sympy_str(expr, expected_str):
    out_str = SympyStrPrinter().doprint(expr)
    assert out_str == expected_str


@pytest.mark.parametrize("expr,expected_str", [
    (symbols('alpha'),
        r'\alpha'),
    (symbols("alpha_1"),
        r'\alpha_{1}'),
    (symbols("alpha_1")**2,
        r'\alpha_{1}^{2}'),
    (symbols("alpha_1")**(symbols('n') + 1),
        r'\alpha_{1}^{n + 1}'),
    (symbols("alpha_1")**((symbols('n') + 1) / 2),
        r'\alpha_{1}^{\frac{n}{2} + \frac{1}{2}}'),
    (I * symbols("alpha_1"),
        r'i \alpha_{1}'),
    (IndexedBase('alpha')[IdxSym('i'), IdxSym('j')],
        r'\alpha_{i j}'),
    (IndexedBase('alpha')[IdxSym('mu'), IdxSym('nu')],
        r'\alpha_{\mu \nu}'),
    (IndexedBase('alpha')[1, 10],
        r'\alpha_{1,10}'),
    (sqrt(2),
        r'\sqrt{2}'),
    (3/sqrt(2),
        r'\frac{3}{\sqrt{2}}'),
    (symbols("alpha_1") / sqrt(2),
        r'\frac{\alpha_{1}}{\sqrt{2}}'),
    (I * symbols("alpha_1") / sqrt(2),
        r'\frac{i \alpha_{1}}{\sqrt{2}}'),
    ((symbols('x') + I * symbols('y')) / sqrt(2),
        r'\frac{x + i y}{\sqrt{2}}'),
    (sqrt(symbols('mu_1') - symbols('mu_2')) * symbols("alpha_1"),
        r'\alpha_{1} \sqrt{\mu_{1} - \mu_{2}}'),
    (exp(-I * symbols("phi")),
        r'e^{- i \phi}'),
    (exp(-I * symbols("phi")) / (1 - symbols('eta_0'))**2,
        r'\frac{e^{- i \phi}}{\left(- \eta_{0} + 1\right)^{2}}'),
    (IdxSym('i'),
        r'i'),
    (IdxSym('alpha'),
        r'\alpha'),
    (IdxSym('alpha_1'),
        r'\alpha_{1}'),
    (IdxSym('alpha', primed=2),
        r"{\alpha^{\prime\prime}}"),
    (sqrt(IdxSym('n')+1),
        r'\sqrt{n + 1}'),
    (sqrt(IdxSym('n', primed=1)+1),
        r'\sqrt{{n^{\prime}} + 1}'),
    (IdxSym('n', primed=1)**2,
        r'{n^{\prime}}^{2}'),
])
def test_sympy_latex(expr, expected_str):
    out_str = SympyLatexPrinter().doprint(expr)
    assert out_str == expected_str
