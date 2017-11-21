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
"""Unicode Printer"""
import re

from sympy.printing.pretty.pretty_symbology import modifier_dict
from sympy.printing.conventions import split_super_sub

from .asciimod import QnetAsciiPrinter
from .sympy import SympyUnicodePrinter


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
        'show_hilbert_space': True,  # alternatively: False, 'subscript'
        'local_sigma_as_ketbra': True,
        'unicode_sub_super': True,
        'operator_hats': True,
    }
    # TODO: allow to drop '*' from products

    _dagger_sym = '†'
    _tensor_sym = '⊗'
    _circuit_series_sym = '◁'
    _circuit_concat_sym = '⊞'

    def _render_str(self, string):
        """Returned a unicodified version of the string"""
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
                'subscript': sub_sup_fmt('|{label}', sub='({space})'),
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
        hs_setting = bool(self._settings['show_hilbert_space'])
        if self._settings['show_hilbert_space'] == 'subscript':
            hs_setting = 'subscript'
        return mapping[expr_type][hs_setting]

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
        if hs is not None and self._settings['show_hilbert_space']:
            hs_label = self._render_hs_label(hs)
        name, total_subscript, total_superscript, args_str \
            = self._split_op(identifier, hs_label, dagger, args)
        if self._settings['operator_hats'] and len(name) == 1:
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

    def _print_IdentitySuperOperator(self, expr):
        return "𝟙"


def unicode(expr, cache=None, **settings):
    """Return a unicode textual representation of the given object /
    expression"""
    try:
        if cache is None and len(settings) == 0:
            return unicode.printer.doprint(expr)
        else:
            printer = unicode._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        unicode._printer_cls = QnetUnicodePrinter
        unicode.printer = unicode._printer_cls()
        return unicode(expr, cache, **settings)


_GREEK_DICTIONARY = {
    'Alpha': 'Α', 'Beta': 'Β', 'Gamma': r'Γ', 'Delta': r'Δ', 'Epsilon': 'Ε',
    'Zeta': 'Ζ', 'Eta': 'Η', 'Theta': r'Τ', 'Iota': 'Ι', 'Kappa': 'Κ',
    'Lambda': r'Λ', 'Mu': 'Μ', 'Nu': 'Ν', 'Xi': r'Ξ', 'Omicron': 'Ο',
    'Pi': r'Π', 'Rho': 'Ρ', 'Sigma': r'Σ', 'Tau': 'Τ', 'Upsilon': r'Υ',
    'Ypsilon': r'Υ', 'Phi': r'Φ', 'Chi': 'Χ', 'Psi': r'Ψ', 'Omega': r'Ω',
    'alpha': 'α', 'beta': 'β', 'gamma': r'γ', 'delta': r'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': r'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': r'λ', 'mu': 'μ', 'nu': 'ν', 'xi': r'ξ', 'omicron': 'ο',
    'pi': r'π', 'rho': 'ρ', 'sigma': r'σ', 'tau': 'τ', 'upsilon': r'υ',
    'ypsilon': r'υ', 'phi': r'φ', 'chi': 'χ', 'psi': r'Ψ', 'omega': r'ω',
    'khi': r'χ', 'Khi': r'Χ', 'varepsilon': r'ε', 'varkappa': r'κ',
    'varphi': r'φ', 'varpi': r'π', 'varrho': r'ρ', 'varsigma': r'ς',
    'vartheta': r'θ',
}

_SUBSCRIPT_MAPPING = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
    '7': '₇', '8': '₈', '9': '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋',
    '=': '₌', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ', 'k': 'ₖ',
    'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
    'β': 'ᵦ', 'γ': 'ᵧ', 'ρ': 'ᵨ', 'φ': 'ᵩ', 'χ': 'ᵪ'
}


_SUPERSCRIPT_MAPPING = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶',
    '7': '⁷', '8': '⁸', '9': '⁹', '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽',
    ')': '⁾', 'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ', 'f': 'ᶠ',
    'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ', 'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ',
    'n': 'ⁿ', 'o': 'ᵒ', 'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
    'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ', 'A': 'ᴬ', 'B': 'ᴮ',
    'D': 'ᴰ', 'E': 'ᴱ', 'G': 'ᴳ', 'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ', 'K': 'ᴷ',
    'L': 'ᴸ', 'M': 'ᴹ', 'N': 'ᴺ', 'O': 'ᴼ', 'P': 'ᴾ', 'R': 'ᴿ', 'T': 'ᵀ',
    'U': 'ᵁ', 'V': 'ⱽ', 'W': 'ᵂ', 'β': 'ᵝ', 'γ': 'ᵞ', 'δ': 'ᵟ', 'ε': 'ᵋ',
    'θ': 'ᶿ', 'ι': 'ᶥ', 'φ': 'ᵠ', 'χ': 'ᵡ'
}


def render_unicode_sub_super(
        name, subs=None, supers=None, sub_first=True, translate_symbols=True,
        unicode_sub_super=True, sep=',', subscript_max_len=1):
    """Assemble a string from the primary name and the given sub- and
    superscripts::

    >>> render_unicode_sub_super(name='alpha', subs=['mu', 'nu'], supers=[2])
    'α_μ,ν^2'

    >>> render_unicode_sub_super(
    ...     name='alpha', subs=['1', '2'], supers=['(1)'], sep='')
    'α₁₂⁽¹⁾'

    >>> render_unicode_sub_super(
    ...     name='alpha', subs=['1', '2'], supers=['(1)'], sep='',
    ...     unicode_sub_super=False)
    'α_12^(1)'

    Args:
        name (str):  the string without the subscript/superscript
        subs (list or None): list of subscripts
        supers (list or None): list of superscripts
        translate_symbols (bool): If True, try to translate (Greek) symbols in
            `name, `subs`, and `supers` to unicode
        unicode_sub_super (bool): It True, try to use unicode
            subscript/superscript symbols
        sep (str): Separator to use if there are multiple
            subscripts/superscripts
        subscript_max_len (int): Maximum character length of subscript that is
            eligible to be rendered in unicode. This defaults to 1, because
            spelling out enire words as a unicode subscript looks terrible in
            monospace (letter spacing too large)
    """
    if subs is None:
        subs = []
    if supers is None:
        supers = []
    if translate_symbols:
        supers = [_translate_symbols(sup) for sup in supers]
        subs = [_translate_symbols(sub) for sub in subs]
        name = _translate_symbols(name)
    res = name
    try:
        if unicode_sub_super:
            supers_modified = [
                    _unicode_sub_super(s, _SUPERSCRIPT_MAPPING)
                    for s in supers]
            subs_modified = [
                    _unicode_sub_super(
                        s, _SUBSCRIPT_MAPPING, max_len=subscript_max_len)
                    for s in subs]
            if sub_first:
                if len(subs_modified) > 0:
                    res += sep.join(subs_modified)
                if len(supers_modified) > 0:
                    res += sep.join(supers_modified)
            else:
                if len(supers_modified) > 0:
                    res += sep.join(supers_modified)
                if len(subs_modified) > 0:
                    res += sep.join(subs_modified)
    except KeyError:
        unicode_sub_super = False
    if not unicode_sub_super:
        sub = sep.join(subs)
        sup = sep.join(supers)
        if sub_first:
            if len(sub) > 0:
                res += "_%s" % sub
            if len(sup) > 0:
                res += "^%s" % sup
        else:
            if len(sup) > 0:
                res += "^%s" % sup
            if len(sub) > 0:
                res += "_%s" % sub
    return res


def _unicode_sub_super(string, mapping, max_len=None):
    """Try to render a subscript or superscript string in unicode, fall back on
    ascii if this is not possible"""
    string = str(string)
    if max_len is not None:
        if len(string) > max_len:
            raise KeyError("max_len exceeded")
    unicode_letters = []
    for letter in string:
        unicode_letters.append(mapping[letter])
    return ''.join(unicode_letters)


def _translate_symbols(string):
    """Given a description of a Greek letter or other special character,
    return the appropriate unicode letter."""
    res = []
    string = str(string)
    for s in re.split(r'(\W+)', string, flags=re.UNICODE):
        tex_str = _GREEK_DICTIONARY.get(s)
        if tex_str:
            res.append(tex_str)
        elif s.lower() in _GREEK_DICTIONARY:
            res.append(_GREEK_DICTIONARY[s])
        else:
            res.append(s)
    return "".join(res)
