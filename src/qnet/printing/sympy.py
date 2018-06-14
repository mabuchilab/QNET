r'''Custom Printers for Sympy expressions

These classes are used by default by the QNET printing systems as sub-printers
for SymPy objects (e.g. for symbolic coefficients). They fix some issues with
SymPy's builtin printers:

* factors like $\frac{1}{\sqrt{2}}$ occur very commonly in quantum mechanics,
  and it is standard notation to write them as such. SymPy insists on
  rationalizing denominators, using $\frac{\sqrt{2}}{2}$ instead. Our custom
  printers restore the canonical form. Note that internally, Sympy still uses
  the rationalized structure; but in any case, Sympy makes no guarantees
  between the algebraic structure of an expression and how it is printed.
* Symbols (especially greek letters) are extremely common, and it's much more
  readable if the string representation of an expression uses unicode for
  these. SymPy supports unicode "pretty-printing"
  (:func:`sympy.printing.pretty.pretty.pretty_print`) only in "2D", where
  expressions are rendered as multiline unicode strings. While this is fine for
  interactive display, it does not work so well for a simple ``str``. The
  :class:`SympyUnicodePrinter` solves this by producing simple strings with
  unicode symbols.
* Some algebraic structures such as factorials, complex-conjugates and indexed
  symbols have sub-optimal rendering in :class:`sympy.printing.str.StrPrinter`
* QNET contains some custom subclasses of SymPy objects (e.g.
  :class:`.IdxSym`) that the default printers don't know how to deal with
  (respectively, render incorrectly!)
'''
from collections import OrderedDict

import sympy
from sympy import sqrt
from sympy.core import S, Rational, Mul, Pow
from sympy.core.mul import _keep_coeff
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter
from sympy.printing.repr import ReprPrinter
from sympy.printing.pretty.pretty_symbology import pretty_symbol

from ._unicode_mappings import _SUPERSCRIPT_MAPPING, _SUBSCRIPT_MAPPING

__all__ = []
__private__ = [
    'SympyLatexPrinter', 'SympyStrPrinter', 'SympyUnicodePrinter',
    'SympyReprPrinter', 'derationalize_denom']


delattr(sympy.Indexed, '_sympystr')  # not sure how else to override printer


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
    """Variation of sympy StrPrinter that derationalizes denominators.

    Additionally, it contains the following modifications:

    * Support for :class:`.IdxSym`
    * Rendering of :class:`sympy.tensor.indexed.Indexed` as subscripts
    * Rendering of :class:`sympy.functions.combinatorial.factorials.factorial`
      as ``!``
    * Option `conjg_style` to configure how complex conjugates are rendered:
      ``'func' renders it as ``conjugate(...)``, and ``'star'`` uses an
      exponentiated asterisk
    """

    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "conjg_style": 'func',
    }

    # _print_IdxSym(self, expr) is implemented in IdxSym._sympystr

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
        indices = [self._print(i) for i in expr.indices]
        sep = ','
        if all([len(i.replace("'", '')) == 1 for i in indices]):
            sep = ''
        return self._print(expr.base)+'_%s' % sep.join(indices)

    def _print_factorial(self, expr, exp=None):
        res = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (res, exp)
        else:
            return res

    def _print_conjugate(self, expr, exp=None):
        if self._settings['conjg_style'] == 'star':
            res = (
                self.parenthesize(expr.args[0], PRECEDENCE["Func"]) + '^*')
        elif self._settings['conjg_style'] in ['func', 'overbar']:
            # recognizing "overbar" is just for compatibility with the other
            # printers
            res = (
                r'conjugate(' + self._print(expr.args[0]) + r')')
            pass
        else:
            raise ValueError(
                "The 'conjg_style' setting must be one of "
                "'star', 'func'")
        if exp is not None:
            return r"%s^%s" % (res, exp)
        else:
            return res


class SympyLatexPrinter(LatexPrinter):
    """Variation of sympy LatexPrinter that derationalizes denominators

    Additionally, it contains the following modifications:

    * Support for :class:`.IdxSym`
    * A setting `conjg_style` that allows to specify how complex conjugate are
      rendered: ``'overline'`` (the default) draws a line over the number,
      'star' uses an exponentiated asterisk, and 'func' renders a a
      ``conjugate`` function
    """

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
        "conjg_style": 'overline',
    }

    # _print_IdxSym(self, expr) is implemented in IdxSym._latex

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

    def _print_conjugate(self, expr, exp=None):
        if self._settings['conjg_style'] == 'overline':
            tex = r"\overline{%s}" % self._print(expr.args[0])
        elif self._settings['conjg_style'] == 'star':
            tex = r"{%s}^*" % self.parenthesize(
                expr.args[0], PRECEDENCE["Func"])
        elif self._settings['conjg_style'] == 'func':
            tex = (
                r'\operatorname{conjugate}\left(' +
                self._print(expr.args[0]) + r'\right)')
        else:
            raise ValueError(
                "The 'conjg_style' setting must be one of "
                "'overline', 'star', 'func'")

        if exp is not None:
            return r"{%s}^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Indexed(self, expr):
        from qnet.printing.latexprinter import _TEX_SINGLE_LETTER_SYMBOLS
        indices = [self._print(i) for i in expr.indices]
        sep = ','
        all_indices_are_single_letters = True
        for i in indices:
            i = i.replace(r'\prime', '')
            if (i not in _TEX_SINGLE_LETTER_SYMBOLS) and (len(i) > 1):
                all_indices_are_single_letters = False
        if all_indices_are_single_letters:
            sep = ' '
        return self._print(expr.base)+'_{%s}' % sep.join(indices)


class SympyUnicodePrinter(SympyStrPrinter):
    """Printer that represents SymPy expressions as (single-line) unicode
    strings.

    This is a mixture of :class:`.StrPrinter`
    and :class:`sympy.printing.pretty.pretty.PrettyPrinter` (minus the 2D
    printing), with the same extensions as :class:`SympyStrPrinter`
    """

    printmethod = "_sympystr"
    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "superscript_asterisk_sym": "\u00A0\u20F0",
        "conjg_style": 'star',
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
        indices = [self._print(i) for i in expr.indices]
        sep = ','
        if all([len(i.replace("'", '')) == 1 for i in indices]):
            sep = ''
        subscript = sep.join(indices)
        try:
            subscript = ''.join([_SUBSCRIPT_MAPPING[l] for l in subscript])
            return self._print(expr.base) + subscript
        except KeyError:
            return self._print(expr.base) + '_%s' % subscript

    def _print_conjugate(self, expr, exp=None):
        if self._settings['conjg_style'] == 'star':
            rendered_arg = self.parenthesize(expr.args[0], PRECEDENCE["Func"])
            if '_' in rendered_arg or '^' in rendered_arg:
                if not rendered_arg.endswith(')'):
                    rendered_arg = "(%s)" % rendered_arg
            res = (rendered_arg + self._settings['superscript_asterisk_sym'])
        elif self._settings['conjg_style'] in ['func', 'overbar']:
            # recognizing "overbar" is just for compatibility with the other
            # printers
            res = (
                r'conjugate(' + self._print(expr.args[0]) + r')')
            pass
        else:
            raise ValueError(
                "The 'conjg_style' setting must be one of "
                "'star', 'func'")
        if exp is not None:
            return r"%s^%s" % (res, exp)
        else:
            return res


class SympyReprPrinter(ReprPrinter):
    """Representation printer with support for
    :class:`.IdxSym`"""

    # _print_IdxSym(self, expr) is implemented in IdxSym._sympyrepr

    def _print_Symbol(self, expr):
        d = expr._assumptions.generator

        d = OrderedDict([(key, d[key]) for key in sorted(d.keys())])
        # the use of an OrderedDict is the only diffference between this and
        # ReprPrinter._print_Symbol. It ensures that we always get the same
        # repr

        # print the dummy_index like it was an assumption
        if expr.is_Dummy:
            d['dummy_index'] = expr.dummy_index

        if d == {}:
            return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s)" % (expr.__class__.__name__,
                                   self._print(expr.name), ', '.join(attr))

        res = self._print_Symbol(expr)
        if expr.primed > 0:
            res = res[:-1] + ", primed=%d)" % expr.primed
        return res
