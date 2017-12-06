#    This file is part of QNET.
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
#############################################################################
r'''Custom Printers for Sympy expressions

These classes are used by default by the QNET printing systems as sub-printers
for SymPy objects (e.g. for symbolic coefficients). They fix some issues with
SymPy's builtin printers:

* factors like $\frac{1}{\sqrt{2}}$ occur very commonly in quantum mechanics,
  and it is standard notation to write them as such. SymPy insists on
  rationalizing denominators, using $\frac{\sqrt{2}}{2}$ instead. Our custom
  printers restore the canonical form. Note that internally, Sympy still uses
  the rationalized structure; in any case, Sympy makes no guarantees between
  the algebraic structure of an expression and how it is printed.
* Symbols (especially greek letters) are extremely common, and it's much more
  readable if the string representation of an expression uses unicode for
  these. SymPy supports unicode "pretty-printing"
  (:func:`sympy.printing.pretty.pretty.pretty_print`) only in "2D", where
  expressions are rendered as multiline unicode strings. While this is fine for
  interactive display, it does not work so well for a simple ``str``. The
  :class:`SympyUnicodePrinter` solves this by producing simple strings with
  unicode symbols.
'''
import sympy
from sympy import sqrt
from sympy.core import S, Rational, Mul, Pow
from sympy.core.mul import _keep_coeff
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty_symbology import pretty_symbol

from ._unicode_mappings import _SUPERSCRIPT_MAPPING, _SUBSCRIPT_MAPPING

__all__ = ['SympyLatexPrinter', 'SympyStrPrinter', 'SympyUnicodePrinter']


delattr(sympy.Indexed, '_sympystr')


def derationalize_denom(expr):
    """Try to de-rationalize the denominator of the given expression.

    The purpose is to allow to reconstruct e.g. ``1/sqrt(2)`` from
    ``sqrt(2)/2``.

    Specifically, this matches `expr` against the following pattern::

        Mul(..., Rational(n, d), Pow(d, Rational(1, 2)), ...)

    and returns a tuple ``(numerator, denom_sq, post_factor)``, where
    ``numerator`` and ``denom_sq`` are ``n`` and ``d`` in the above pattern (of
    type `int`), respectively, and ``post_factor`` is the product of the
    remaining factors (``...`` in `expr`). The result will fulfill the
    following identity::

        (numerator / sqrt(denom_sq)) * post_factor == expr

    If `expr` does not follow the appropriate pattern, a :exc:`ValueError` is
    raised.
    """
    r_pos = -1
    p_pos = -1
    numerator = S.Zero
    denom_sq = S.One
    post_factors = []
    if isinstance(expr, Mul):
        for pos, factor in enumerate(expr.args):
            if isinstance(factor, Rational) and r_pos < 0:
                r_pos = pos
                numerator, denom_sq = factor.p, factor.q
            elif isinstance(factor, Pow) and r_pos >= 0:
                if factor == sqrt(denom_sq):
                    p_pos = pos
                else:
                    post_factors.append(factor)
            else:
                post_factors.append(factor)
        if r_pos >= 0 and p_pos >= 0:
            return numerator, denom_sq, Mul(*post_factors)
        else:
            raise ValueError("Cannot derationalize")
    else:
        raise ValueError("expr is not a Mul instance")


class SympyStrPrinter(StrPrinter):
    """Variation of sympy StrPrinter that derationalizes denominators"""

    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
    }

    def _print_Mul(self, expr):

        prec = precedence(expr)

        try:
            numerator, denom_sq, post_factor = derationalize_denom(expr)
            if post_factor == S.One:
                return "%s/sqrt(%s)" % (numerator, denom_sq)
            else:
                if numerator == 1:
                    return "%s / sqrt(%s)" % (
                        self.parenthesize(post_factor, prec), denom_sq)
                else:
                    return "(%s/sqrt(%s)) %s" % (
                        numerator, denom_sq,
                        self.parenthesize(post_factor, prec))
        except ValueError:
            return super()._print_Mul(expr)

    def _print_Indexed(self, expr):
        return self._print(expr.base)+'_%s' % ','.join(
            map(self._print, expr.indices))


class SympyLatexPrinter(LatexPrinter):
    """Variation of sympy LatexPrinter that derationalizes denominators"""

    printmethod = "_latex"

    _default_settings = {
        "order": None,
        "mode": "plain",
        "itex": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "long_frac_ratio": 2,
        "mul_symbol": None,
        "inv_trig_style": "abbreviated",
        "mat_str": None,
        "mat_delim": "[",
        "symbol_names": {},
    }

    def _print_Mul(self, expr):

        prec = precedence(expr)

        try:
            numerator, denom_sq, post_factor = derationalize_denom(expr)
            if post_factor == S.One:
                return r'\frac{%s}{\sqrt{%s}}' % (numerator, denom_sq)
            else:
                if numerator == 1:
                    return r'\frac{%s}{\sqrt{%s}}' % (
                        self._print(post_factor), denom_sq)
                else:
                    return r'\frac{%s}{\sqrt{%s}} %s' % (
                        numerator, denom_sq,
                        self.parenthesize(post_factor, prec))
        except ValueError:
            return super()._print_Mul(expr)


class SympyUnicodePrinter(SympyStrPrinter):
    """Printer that represents SymPy expressions as (single-line) unicode
    strings. This is a mixture of the default Sympy StrPrinter and the SymPy
    PrettyPrinter
    """

    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
    }

    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)

        PREC = precedence(expr)
        l = []
        for term in terms:
            t = str(self._print(term))
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < PREC:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_ComplexInfinity(self, expr):
        return '∞'

    def _print_ImaginaryUnit(self, expr):
        return 'ⅈ'

    def _print_Infinity(self, expr):
        return '∞'

    def _print_Inverse(self, I):
        return "%s⁻¹" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Pi(self, expr):
        return 'π'

    def _print_Mul(self, expr):

        prec = precedence(expr)

        try:
            numerator, denom_sq, post_factor = derationalize_denom(expr)
            if post_factor == S.One:
                return "%s/√%s" % (numerator, denom_sq)
            else:
                if numerator == 1:
                    return "%s / √%s" % (
                        self.parenthesize(post_factor, prec), denom_sq)
                else:
                    return "(%s/√%s) %s" % (
                        numerator, denom_sq,
                        self.parenthesize(post_factor, prec))
        except ValueError:
            pass  # Continue below

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and
                    item.exp.is_Rational and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [str(self.parenthesize(x, prec)) for x in a]
        b_str = [str(self.parenthesize(x, prec)) for x in b]

        if len(b) == 0:
            return sign + ' '.join(a_str)
        elif len(b) == 1:
            return sign + ' '.join(a_str) + "/" + b_str[0]
        else:
            return sign + ' '.join(a_str) + "/(%s)" % ' '.join(b_str)

    def _print_NegativeInfinity(self, expr):
        return '-∞'

    def _print_Pow(self, expr, rational=False):
        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "√%s" % self.parenthesize(expr.base, PREC)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that
                # will match -0.5, which we don't want.
                return "1/√%s" % self.parenthesize(expr.base, PREC)
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '1/%s' % self.parenthesize(expr.base, PREC)

        e = self.parenthesize(expr.exp, PREC)
        if (self.printmethod == '_sympyrepr' and
                expr.exp.is_Rational and expr.exp.q != 1):
            # the parenthesized exp should be '(Rational(a, b))' so strip
            # parens, but just check to be sure.
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC), e[1:-1])
        try:
            e_super = ''.join([_SUPERSCRIPT_MAPPING[l] for l in e])
            return '%s%s' % (self.parenthesize(expr.base, PREC), e_super)
        except KeyError:
            return '%s**%s' % (self.parenthesize(expr.base, PREC), e)

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        b = str(self.parenthesize(expr.base, PREC))
        e = str(self.parenthesize(expr.exp, PREC))
        try:
            e_super = ''.join([_SUPERSCRIPT_MAPPING[l] for l in e])
            return '%s%s' % (b, e_super)
        except KeyError:
            return '%s**%s' % (b, e)

    def _print_Symbol(self, e):
        return pretty_symbol(e.name)
    _print_RandomSymbol = _print_Symbol

    def _print_Identity(self, expr):
        return "𝟙"

    def _print_ZeroMatrix(self, expr):
        return "𝟘"

    def _print_Indexed(self, expr):
        subscript = ','.join(map(self._print, expr.indices))
        try:
            subscript = ''.join([_SUBSCRIPT_MAPPING[l] for l in subscript])
            return self._print(expr.base) + subscript
        except KeyError:
            return self._print(expr.base) + '_%s' % subscript
