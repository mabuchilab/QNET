"""
Routines for rendering expressions to LaTeX
"""
import re

from sympy.printing.latex import (greek_letters_set, other_symbols)
from sympy.printing.conventions import split_super_sub
from numpy import complex128

from ..utils.singleton import Singleton
from ..utils.indices import StrLabel
from .asciiprinter import QnetAsciiPrinter
from ._precedence import precedence, PRECEDENCE
from .sympy import SympyLatexPrinter

__all__ = []
__private__ = ['QnetLatexPrinter', 'render_latex_sub_super']


class QnetLatexPrinter(QnetAsciiPrinter):
    """Printer for a LaTeX representation.

    See :func:`qnet.printing.latex` for documentation of `settings`.
    """
    sympy_printer_cls = SympyLatexPrinter
    printmethod = '_latex'

    _default_settings = {  # documented in :func:`latex`
        'show_hs_label': True,  # alternatively: False, 'subscript'
        'sig_as_ketbra': True,
        'tex_op_macro': r'\hat{{{name}}}',
        'tex_textop_macro': r'\text{{{name}}}',
        'tex_sop_macro': r'\mathrm{{{name}}}',
        'tex_textsop_macro': r'\mathrm{{{name}}}',
        'tex_identity_sym': r'\mathbb{1}',
        'tex_use_braket': False,  # use the braket package?
        'tex_frac_for_spin_labels': False,
    }

    _parenth_left = r'\left('
    _parenth_right = r'\right)'
    _bracket_left = r'\left['
    _bracket_right = r'\right]'
    _dagger_sym = r'\dagger'
    _tensor_sym = r'\otimes'
    _product_sym = ' '
    _circuit_series_sym = r'\lhd'
    _circuit_concat_sym = r'\boxplus'
    _cid = r'{\rm cid}(%d)'
    _sum_sym = r'\sum'
    _element_sym = r'\in'
    _ellipsis = r'\dots'
    _set_delim_left = r'\{'
    _set_delim_right = r'\}'

    def __init__(self, cache=None, settings=None):
        super().__init__(cache=cache, settings=settings)
        # enable the cache to provide strings for symbols *inside* SymPy
        # Expressions
        if cache is not None:
            if 'symbol_names' in self._sympy_printer._default_settings:
                self._sympy_printer._settings['symbol_names'] = cache

    def _print_SCALAR_TYPES(self, expr, *args, **kwargs):
        res = super()._print_SCALAR_TYPES(expr, *args, **kwargs)
        if isinstance(expr, (complex, complex128)):
            res = res.replace('j', 'i')
        return res

    @classmethod
    def _is_single_letter(cls, label):
        return (len(label) == 1 or label in _TEX_SINGLE_LETTER_SYMBOLS)

    def _render_str(self, string):
        """Returned a texified version of the string"""
        if isinstance(string, StrLabel):
            string = string._render(string.expr)
        string = str(string)
        if len(string) == 0:
            return ''
        name, supers, subs = split_super_sub(string)
        return render_latex_sub_super(
            name, subs, supers, translate_symbols=True)

    def _render_hs_label(self, hs):
        """Return the label of the given Hilbert space as a string"""
        if isinstance(hs.__class__, Singleton):
            return self._render_str(hs.label)
        else:
            tensor_sym = ' %s ' % self._tensor_sym
            return tensor_sym.join(
                [self._render_str(ls.label) for ls in hs.local_factors])

    def _braket_fmt(self, expr_type):
        """Return a format string for printing an `expr_type`
        ket/bra/ketbra/braket"""
        mapping = {
            True: {  # use braket package
                'bra': {
                    True: r'\Bra{{{label}}}^{{({space})}}',
                    'subscript': r'\Bra{{{label}}}_{{({space})}}',
                    False:  r'\Bra{{{label}}}'},
                'ket': {
                    True: r'\Ket{{{label}}}^{{({space})}}',
                    'subscript': r'\Ket{{{label}}}_{{({space})}}',
                    False:  r'\Ket{{{label}}}'},
                'ketbra': {
                    True:
                        r'\Ket{{{label_i}}}\!\Bra{{{label_j}}}^{{({space})}}',
                    'subscript':
                        r'\Ket{{{label_i}}}\!\Bra{{{label_j}}}_{{({space})}}',
                    False: r'\Ket{{{label_i}}}\!\Bra{{{label_j}}}'},
                'braket': {
                    True: r'\Braket{{{label_i} | {label_j}}}^({space})',
                    'subscript': r'\Braket{{{label_i} | {label_j}}}_({space})',
                    False:  r'\Braket{{{label_i} | {label_j}}}'}},
            False: {  # explicit tex macros
                'bra': {
                    True:
                        r'\left\langle {label} \right\rvert^{{({space})}}',
                    'subscript':
                        r'\left\langle {label} \right\rvert^{{({space})}}',
                    False:
                        r'\left\langle {label} \right\rvert'},
                'ket': {
                    True:
                        r'\left\lvert {label} \right\rangle^{{({space})}}',
                    'subscript':
                        r'\left\lvert {label} \right\rangle_{{({space})}}',
                    False:  r'\left\lvert {label} \right\rangle'},
                'ketbra': {
                    True:
                        r'\left\lvert {label_i} \middle\rangle\!'
                        r'\middle\langle {label_j} \right\rvert^{{({space})}}',
                    'subscript':
                        r'\left\lvert {label_i} \middle\rangle\!'
                        r'\middle\langle {label_j} \right\rvert_{{({space})}}',
                    False:
                        r'\left\lvert {label_i} \middle\rangle\!'
                        r'\middle\langle {label_j} \right\rvert'},
                'braket': {
                    True:
                        r'\left\langle {label_i} \middle\vert '
                        r'{label_j} \right\rangle^{{({space})}}',
                    'subscript':
                        r'\left\langle {label_i} \middle\vert '
                        r'{label_j} \right\rangle_{{({space})}}',
                    False:
                        r'\left\langle {label_i} \middle\vert '
                        r'{label_j} \right\rangle'}}
            }
        hs_setting = bool(self._settings['show_hs_label'])
        if self._settings['show_hs_label'] == 'subscript':
            hs_setting = 'subscript'
        return mapping[self._settings['tex_use_braket']][expr_type][hs_setting]

    def _render_op(
            self, identifier, hs=None, dagger=False, args=None, superop=False):
        """Render an operator

        Args:
            identifier (str): The identifier (name/symbol) of the operator. May
                include a subscript, denoted by '_'.
            hs (qnet.algebra.hilbert_space_algebra.HilbertSpace): The Hilbert
                space in which the operator is defined
            dagger (bool): Whether the operator should be daggered
            args (list): A list of expressions that will be rendered with
                :meth:`doprint`, joined with commas, enclosed in parenthesis
            superop (bool): Whether the operator is a super-operator
        """
        hs_label = None
        if hs is not None and self._settings['show_hs_label']:
            hs_label = self._render_hs_label(hs)
        name, total_subscript, total_superscript, args_str \
            = self._split_op(identifier, hs_label, dagger, args)
        if name.startswith(r'\text{'):
            name = name[6:-1]
        if self._is_single_letter(name):
            if superop:
                name_fmt = self._settings['tex_sop_macro']
            else:
                name_fmt = self._settings['tex_op_macro']
        else:
            if superop:
                name_fmt = self._settings['tex_textsop_macro']
            else:
                name_fmt = self._settings['tex_textop_macro']
        res = name_fmt.format(name=name)
        res = render_latex_sub_super(
            res, [total_subscript], [total_superscript],
            translate_symbols=True)
        res += args_str
        return res

    def _render_state_label(self, label):
        if self._isinstance(label, 'SymbolicLabelBase'):
            return self._print_SCALAR_TYPES(label.expr)
        else:
            label = self._render_str(label)
            if "/" in label and self._settings['tex_frac_for_spin_labels']:
                numer, denom = label.split("/", 1)
                sign = '+'
                if numer.startswith('-') or numer.startswith('+'):
                    sign, numer = numer[0], numer[1:]
                return r'%s\frac{%s}{%s}' % (sign, numer, denom)
            else:
                return label

    def _print_Commutator(self, expr, adjoint=False):
        res = (
            r'\left[' + self.doprint(expr.A) + ", " + self.doprint(expr.B) +
            r'\right]')
        if adjoint:
            res += "^{%s}" % self._dagger_sym
        return res

    def _print_OperatorTrace(self, expr, adjoint=False):
        s = self._render_hs_label(expr._over_space)
        kwargs = {}
        if adjoint:
            kwargs['adjoint'] = adjoint
        o = self.doprint(expr.operand, **kwargs)
        return r'{{\rm tr}}_{{{space}}}\left[{operand}\right]'.format(
            space=s, operand=o)

    def _print_Feedback(self, expr):
        operand = self.doprint(expr.operand)
        o, i = expr.out_in_pair
        return r'\left\lfloor{%s}\right\rfloor_{%s\rightarrow{}%s}' % (
            operand, o, i)

    def _print_SeriesInverse(self, expr):
        return r'\left[%s\right]^{\rhd}' % self.doprint(expr.operand)

    def _print_CircuitSymbol(self, expr):
        res = self._render_str(expr.label)
        if len(expr.sym_args) > 0:
            res += (
                self._parenth_left +
                ", ".join([self.doprint(arg) for arg in expr.sym_args]) +
                self._parenth_right)
        return res

    def _print_Component(self, expr):
        res = r'{\rm %s}' % self._render_str(expr.label)
        if len(expr.sym_args) > 0:
            res += (
                self._parenth_left +
                ", ".join([self.doprint(arg) for arg in expr.sym_args]) +
                self._parenth_right)
        return res

    def _print_CPermutation(self, expr):
        permutation_sym = r'\mathbf{P}_{\sigma}'
        return r'%s\begin{pmatrix} %s \\ %s \end{pmatrix}' % (
            permutation_sym,
            " & ".join(map(str, range(expr.cdim))),
            " & ".join(map(str, expr.permutation)))

    def _print_CIdentity(self, expr):
        return r'{\rm cid}(1)'

    def _print_CircuitZero(self, expr):
        return r'{\rm cid}(0)'

    def _print_HilbertSpace(self, expr):
        return r'\mathcal{{H}}_{{{label}}}'.format(
            label=self._render_hs_label(expr))

    def _print_IdentityOperator(self, expr):
        return self._settings['tex_identity_sym']

    def _print_ZeroOperator(self, expr):
        return r'\mathbb{0}'

    def _print_ZeroSuperOperator(self, expr):
        return r'\mathbb{0}'

    def _print_IdentitySuperOperator(self, expr):
        return self._settings['tex_identity_sym']

    def _print_SPre(self, expr, superop=True):
        name = self._settings['tex_textsop_macro'].format(name='SPre')
        return (
            name + self._parenth_left + self.doprint(expr.operands[0]) +
            self._parenth_right)

    def _print_SPost(self, expr, superop=True):
        name = self._settings['tex_textsop_macro'].format(name='SPost')
        return (
            name + self._parenth_left + self.doprint(expr.operands[0]) +
            self._parenth_right)

    def _print_SuperOperatorTimesOperator(self, expr):
        prec = precedence(expr)
        sop, op = expr.sop, expr.op
        cs = self.parenthesize(sop, prec)
        ct = self.doprint(op)
        return r'%s\left[%s\right]' % (cs, ct)

    def _print_QuantumDerivative(self, expr):
        res = ""
        numerator = r'\partial'
        if expr.n > 1:
            numerator = r'\partial^{%s}' % expr.n
        denominators = []
        for sym, n in expr.derivs.items():
            if n == 1:
                denominators.append(
                    r'\partial %s' % self.doprint(sym))
            else:
                denominators.append(
                    r'\partial %s^{%s}' % (self.doprint(sym), n))
        denominator = " ".join(denominators)
        res += r'\frac{%s}{%s}' % (numerator, denominator)
        res += " " + self.parenthesize(
            expr.operand, PRECEDENCE['Mul'], strict=True)
        if expr.vals:
            res = r'\left. ' + res
            evaluation_strs = []
            for sym, val in expr.vals.items():
                evaluation_strs.append(
                    "%s=%s" % (self.doprint(sym), self.doprint(val)))
            evaluation_str = ", ".join(evaluation_strs)
            res += r' \right\vert_{%s}' % evaluation_str
        return res

    def _print_Matrix(self, expr):
        matrix_left_sym = r'\begin{pmatrix}'
        matrix_right_sym = r'\end{pmatrix}'
        matrix_row_left_sym = r''
        matrix_row_right_sym = r''
        matrix_col_sep_sym = r' & '
        matrix_row_sep_sym = r' \\'
        row_strs = []
        if len(expr.matrix) == 0:
            row_strs.append(matrix_row_left_sym + matrix_row_right_sym)
            row_strs.append(matrix_row_left_sym + matrix_row_right_sym)
        else:
            for row in expr.matrix:
                row_strs.append(
                    matrix_row_left_sym +
                    matrix_col_sep_sym.join(
                        [self.doprint(entry) for entry in row]) +
                    matrix_row_right_sym)
        return (
            matrix_left_sym + matrix_row_sep_sym.join(row_strs) +
            matrix_right_sym)

    def _print_Eq(self, expr):
        # print for qnet.algebra.toolbox.equality.Eq, but also works for any
        # Eq class that has the minimum requirement to have an `lhs` and `rhs`
        # attribute
        try:
            has_history = len(expr._prev_rhs) > 0
        except AttributeError:
            has_history = False
        if has_history:
            res = r'\begin{align}' + "\n"
            res += "  %s &= %s" % (
                self.doprint(expr._prev_lhs[0]),
                self.doprint(expr._prev_rhs[0]))
            if expr._prev_tags[0] is not None:
                res += r'\tag{%s}' % expr._prev_tags[0]
            res += "\\\\\n"
            for i, rhs in enumerate(expr._prev_rhs[1:]):
                lhs = expr._prev_lhs[i+1]
                if lhs is None:
                    res += "   &= %s" % self.doprint(rhs)
                else:
                    res += "  %s &= %s" % (self.doprint(lhs), self.doprint(rhs))
                if expr._prev_tags[i+1] is not None:
                    res += r'\tag{%s}' % expr._prev_tags[i+1]
                res += "\\\\\n"
            lhs = expr._lhs
            if lhs is None:
                res += "   &= %s\n" % self.doprint(expr.rhs)
            else:
                res += "  %s &= %s\n" % (
                    self.doprint(lhs), self.doprint(expr.rhs))
            if expr.tag is not None:
                res += r'\tag{%s}' % expr.tag
            res += r'\end{align}' + "\n"
        else:
            res = r'\begin{equation}' + "\n"
            res += "  %s = %s\n" % (
                self.doprint(expr.lhs), self.doprint(expr.rhs))
            try:
                if expr.tag is not None:
                    res += r'\tag{%s}' % expr.tag
            except AttributeError:
                pass
            res += r'\end{equation}' + "\n"
        return res


_TEX_GREEK_DICTIONARY = {
    'Alpha': 'A', 'Beta': 'B', 'Gamma': r'\Gamma', 'Delta': r'\Delta',
    'Epsilon': 'E', 'Zeta': 'Z', 'Eta': 'H', 'Theta': r'\Theta', 'Iota': 'I',
    'Kappa': 'K', 'Lambda': r'\Lambda', 'Mu': 'M', 'Nu': 'N', 'Xi': r'\Xi',
    'omicron': 'o', 'Omicron': 'O', 'Pi': r'\Pi', 'Rho': 'P',
    'Sigma': r'\Sigma', 'Tau': 'T', 'Upsilon': r'\Upsilon',
    'Ypsilon': r'\Upsilon', 'ypsilon': r'\upsilon', 'upsilon': r'\upsilon',
    'Phi': r'\Phi', 'Chi': 'X', 'Psi': r'\Psi', 'Omega': r'\Omega',
    'lamda': r'\lambda', 'Lamda': r'\Lambda', 'khi': r'\chi',
    'Khi': r'X', 'varepsilon': r'\varepsilon', 'varkappa': r'\varkappa',
    'varphi': r'\varphi', 'varpi': r'\varpi', 'varrho': r'\varrho',
    'varsigma': r'\varsigma', 'vartheta': r'\vartheta', 'up': r'\uparrow',
    'down': r'\downarrow', 'uparrow': r'\uparrow', 'downarrow': r'\downarrow',
}


_TEX_SINGLE_LETTER_SYMBOLS = [
    r'\Delta', r'\Gamma', r'\Lambda', r'\Omega', r'\Phi', r'\Pi', r'\Psi',
    r'\Sigma', r'\Theta', r'\Upsilon', r'\Xi', r'\alpha', r'\beta', r'\chi',
    r'\delta', r'\epsilon', r'\eta', r'\gamma', r'\iota', r'\kappa',
    r'\lambda', r'\mu', r'\nu', r'\omega', r'\phi', r'\pi', r'\psi', r'\rho',
    r'\sigma', r'\tau', r'\theta', r'\upsilon', r'\varepsilon', r'\varphi',
    r'\varrho', r'\vartheta', r'\xi', r'\zeta', r'\uparrow', r'\downarrow']


def _translate_symbols(string):
    """Given a description of a Greek letter or other special character,
    return the appropriate latex."""
    res = []
    for s in re.split(r'([,.:\s=]+)', string):
        tex_str = _TEX_GREEK_DICTIONARY.get(s)
        if tex_str:
            res.append(tex_str)
        elif s.lower() in greek_letters_set:
            res.append("\\" + s.lower())
        elif s in other_symbols:
            res.append("\\" + s)
        else:
            if re.match(r'^[a-zA-Z]{4,}$', s):
                res.append(r'\text{' + s + '}')
            else:
                res.append(s)
    return "".join(res)


def render_latex_sub_super(
        name, subs=None, supers=None, translate_symbols=True, sep=','):
    r'''Assemble a string from the primary name and the given sub- and
    superscripts::

        >>> render_latex_sub_super(name='alpha', subs=['mu', 'nu'], supers=[2])
        '\\alpha_{\\mu,\\nu}^{2}'

        >>> render_latex_sub_super(
        ...     name='alpha', subs=['1', '2'], supers=['(1)'], sep='')
        '\\alpha_{12}^{(1)}'

    Args:
        name (str):  the string without the subscript/superscript
        subs (list or None): list of subscripts
        supers (list or None): list of superscripts
        translate_symbols (bool): If True, try to translate (Greek) symbols in
            `name, `subs`, and `supers` to unicode
        sep (str): Separator to use if there are multiple
            subscripts/superscripts
    '''
    if subs is None:
        subs = []
    if supers is None:
        supers = []
    if translate_symbols:
        supers = [_translate_symbols(str(sup)) for sup in supers]
        subs = [_translate_symbols(str(sub)) for sub in subs]
        name = _translate_symbols(name)
    res = name
    sub = sep.join(subs)
    sup = sep.join(supers)
    if len(sub) > 0:
        res += "_{%s}" % sub
    if len(sup) > 0:
        res += "^{%s}" % sup
    return res
