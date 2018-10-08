"""Unicode Printer"""
from sympy.printing.pretty.pretty_symbology import modifier_dict
from sympy.printing.conventions import split_super_sub

from ..utils.indices import StrLabel
from .asciiprinter import QnetAsciiPrinter
from .sympy import SympyUnicodePrinter
from ._unicode_mappings import render_unicode_sub_super
from ._precedence import precedence, PRECEDENCE

__all__ = []
__private__ = ['QnetUnicodePrinter', 'SubSupFmt', 'SubSupFmtNoUni']


class SubSupFmt():
    """A format string that divides into a name, subscript, and superscript

    >>> fmt = SubSupFmt('{name}', sub='({i},{j})', sup='({sup})')
    >>> fmt.format(name='alpha', i='mu', j='nu', sup=1)
    'α_(μ,ν)^(1)'
    >>> fmt = SubSupFmt('{name}', sub='{sub}', sup='({sup})')
    >>> fmt.format(name='alpha', sub='1', sup=1)
    'α₁⁽¹⁾'
    """
    def __init__(self, name, sub=None, sup=None, unicode_sub_super=True):
        self.name = name
        self.sub = sub
        self.sup = sup
        self.unicode_sub_super = unicode_sub_super

    def format(self, **kwargs):
        """Format and combine the name, subscript, and superscript"""
        name = self.name.format(**kwargs)

        subs = []
        if self.sub is not None:
            subs = [self.sub.format(**kwargs)]
        supers = []
        if self.sup is not None:
            supers = [self.sup.format(**kwargs)]

        return render_unicode_sub_super(
            name, subs, supers, sub_first=True, translate_symbols=True,
            unicode_sub_super=self.unicode_sub_super)

    def __repr__(self):
        return "%s(%r, sub=%r, sup=%r, unicode_sub_super=%r)" % (
            self.__class__.__name__, self.name, self.sub, self.sup,
            self.unicode_sub_super)


class SubSupFmtNoUni(SubSupFmt):
    """SubSupFmt with default unicode_sub_super=False"""
    def __init__(self, name, sub=None, sup=None, unicode_sub_super=False):
        super().__init__(name, sub, sup, unicode_sub_super)


class QnetUnicodePrinter(QnetAsciiPrinter):
    """Printer for a string (Unicode) representation."""
    sympy_printer_cls = SympyUnicodePrinter
    printmethod = '_unicode'

    _default_settings = {
        'show_hs_label': True,  # alternatively: False, 'subscript'
        'sig_as_ketbra': True,
        'unicode_sub_super': True,
        'unicode_op_hats': True,
    }
    _dagger_sym = '†'
    _tensor_sym = '⊗'
    _product_sym = ' '
    _circuit_series_sym = '◁'
    _circuit_concat_sym = '⊞'
    _sum_sym = '∑'
    _element_sym = '∈'
    _ellipsis = '…'

    def _render_str(self, string):
        """Returned a unicodified version of the string"""
        if isinstance(string, StrLabel):
            string = string._render(string.expr)
        string = str(string)
        if len(string) == 0:
            return ''
        name, supers, subs = split_super_sub(string)
        return render_unicode_sub_super(
            name, subs, supers, sub_first=True, translate_symbols=True,
            unicode_sub_super=self._settings['unicode_sub_super'])

    def _braket_fmt(self, expr_type):
        """Return a format string for printing an `expr_type`
        ket/bra/ketbra/braket"""
        if self._settings['unicode_sub_super']:
            sub_sup_fmt = SubSupFmt
        else:
            sub_sup_fmt = SubSupFmtNoUni
        mapping = {
            'bra': {
                True: sub_sup_fmt('⟨{label}|', sup='({space})'),
                'subscript': sub_sup_fmt('⟨{label}|', sub='({space})'),
                False:  sub_sup_fmt('⟨{label}|')},
            'ket': {
                True: sub_sup_fmt('|{label}⟩', sup='({space})'),
                'subscript': sub_sup_fmt('|{label}⟩', sub='({space})'),
                False:  sub_sup_fmt('|{label}⟩')},
            'ketbra': {
                True: sub_sup_fmt('|{label_i}⟩⟨{label_j}|', sup='({space})'),
                'subscript': sub_sup_fmt(
                    '|{label_i}⟩⟨{label_j}|', sub='({space})'),
                False:  sub_sup_fmt('|{label_i}⟩⟨{label_j}|')},
            'braket': {
                True: sub_sup_fmt('⟨{label_i}|{label_j}⟩', sup='({space})'),
                'subscript': sub_sup_fmt(
                    '⟨{label_i}|{label_j}⟩', sub='({space})'),
                False:  sub_sup_fmt('⟨{label_i}|{label_j}⟩')},
        }
        hs_setting = bool(self._settings['show_hs_label'])
        if self._settings['show_hs_label'] == 'subscript':
            hs_setting = 'subscript'
        return mapping[expr_type][hs_setting]

    def _render_op(
            self, identifier, hs=None, dagger=False, args=None, superop=False):
        """Render an operator

        Args:
            identifier (str or SymbolicLabelBase): The identifier (name/symbol)
                of the operator. May include a subscript, denoted by '_'.
            hs (HilbertSpace): The Hilbert space in which the operator is
                defined
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
        if self._settings['unicode_op_hats'] and len(name) == 1:
            if superop:
                res = name
            else:
                res = modifier_dict['hat'](name)
        else:
            res = name
        res = render_unicode_sub_super(
            res, [total_subscript], [total_superscript], sub_first=True,
            translate_symbols=True,
            unicode_sub_super=self._settings['unicode_sub_super'])
        res += args_str
        return res

    def _print_Feedback(self, expr):
        operand = self.doprint(expr.operand)
        o, i = expr.out_in_pair
        if self._settings['unicode_sub_super']:
            return render_unicode_sub_super(
                '[%s]' % operand, subs=['%s-%s' % (o, i)],
                translate_symbols=False, subscript_max_len=9)
        else:
            return '[%s]_%s→%s' % (operand, o, i)

    def _print_SeriesInverse(self, expr):
        return r'[{operand}]⁻¹'.format(
            operand=self.doprint(expr.operand))

    def _print_HilbertSpace(self, expr):
        return render_unicode_sub_super(
            'ℌ', subs=[self._render_hs_label(expr)])

    def _print_IdentityOperator(self, expr):
        return "𝟙"

    def _print_IdentitySuperOperator(self, expr, superop=True):
        return "𝟙"

    def _print_QuantumDerivative(self, expr):
        res = ""
        for sym, n in expr.derivs.items():
            sym_str = self.doprint(sym)
            if " " in sym_str:
                sym_str = "(%s)" % sym_str
            subs = [sym_str, ]
            if n == 1:
                supers = []
            else:
                supers = [self.doprint(n), ]
            res += (render_unicode_sub_super(
                '∂', subs, supers, sub_first=True, translate_symbols=True,
                unicode_sub_super=self._settings['unicode_sub_super']) + " ")
        res += self.parenthesize(expr.operand, PRECEDENCE['Mul'], strict=True)
        if expr.vals:
            evaluation_strs = []
            for sym, val in expr.vals.items():
                evaluation_strs.append(
                    "%s=%s" % (self.doprint(sym), self.doprint(val)))
            evaluation_str = ", ".join(evaluation_strs)
            if " " in evaluation_str:
                evaluation_str = "(%s)" % evaluation_str
            res += (render_unicode_sub_super(
                ' |', subs=[evaluation_str, ], supers=[],
                sub_first=True, translate_symbols=True,
                unicode_sub_super=self._settings['unicode_sub_super'],
                subscript_max_len=3))
        return res
