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
#Ξ###########################################################################
"""
Routines for rendering expressions to Unicode
"""
import re

from numpy import complex128
import sympy
from sympy.printing.conventions import split_super_sub
from sympy.printing.pretty.pretty_symbology import modifier_dict

from .base import Printer
from ..algebra.singleton import Singleton, singleton_object
from ..algebra.scalar_types import SCALAR_TYPES


class _circuit_fb_fmt():
    @staticmethod
    def format(operand, output, input):
        try:
            return r'[{operand}]{output}₋{input}'.format(
                    operand=operand,
                    output=unicode_sub_super(output, subscript_mapping),
                    input=unicode_sub_super(input, subscript_mapping))
        except KeyError:
            return r'[{operand}]_{output}→{input}'.format(
                    operand=operand, output=output, input=input)


class _hilbert_space_fmt():
    @staticmethod
    def format(label):
        try:
            return r'ℌ{label}'.format(
                    label=unicode_sub_super(label, subscript_mapping, 1))
        except KeyError:
            return r'ℌ_{label}'.format(label=label)


class _bra_fmt():
    @staticmethod
    def format(label, space):
        try:
            return r'⟨{label}|{space}'.format(
                    label=label,
                    space=unicode_sub_super("("+space+")",
                                            subscript_mapping, 3))
        except KeyError:
            return r'⟨{label}|_({space})'.format(label=label, space=space)


class _ket_fmt():
    @staticmethod
    def format(label, space):
        try:
            return r'|{label}⟩{space}'.format(
                    label=label,
                    space=unicode_sub_super("("+space+")",
                                            subscript_mapping, 3))
        except KeyError:
            return r'|{label}⟩_({space})'.format(label=label, space=space)


class _ketbra_fmt():
    @staticmethod
    def format(label_i, label_j, space):
        try:
            return r'|{label_i}⟩⟨{label_j}|{space}'.format(
                    label_i=label_i, label_j=label_j,
                    space=unicode_sub_super("("+space+")",
                                            subscript_mapping, 3))
        except KeyError:
            return r'|{label_i}⟩⟨{label_j}|_({space})'.format(
                    label_i=label_i, label_j=label_j, space=space)


class _braket_fmt():
    @staticmethod
    def format(label_i, label_j, space):
        try:
            return r'⟨{label_i}|{label_j}⟩{space}'.format(
                    label_i=label_i, label_j=label_j,
                    space=unicode_sub_super("("+space+")",
                                            subscript_mapping, 3))
        except KeyError:
            return r'⟨{label_i}|{label_j}⟩_({space})'.format(
                    label_i=label_i, label_j=label_j, space=space)


@singleton_object
class UnicodePrinter(Printer, metaclass=Singleton):
    """Printer that renders greek latters and sub-/superscripts in unicode. See
    :class:`Printer` for details"""

    head_repr_fmt = r'{head}({args}{kwargs})'
    identity_sym = '𝟙'
    zero_sym = '0'
    dagger_sym = r'†'
    daggered_sym = r'^†'
    scalar_product_sym = r'*'
    tensor_sym = r'⊗'
    inner_product_sym = r'·'
    op_product_sym = r' '
    circuit_series_sym = '◁'
    circuit_concat_sym = '⊞'
    circuit_fb_fmt = _circuit_fb_fmt
    null_space_proj_sym = "P_Ker"
    hilbert_space_fmt = _hilbert_space_fmt
    bra_fmt = _bra_fmt
    ket_fmt = _ket_fmt
    ketbra_fmt = _ketbra_fmt
    braket_fmt = _braket_fmt

    _registry = {}

    @classmethod
    def _render(cls, expr, adjoint=False):
        return expr._unicode_(adjoint=adjoint)

    @classmethod
    def render_string(cls, ascii_str):
        """Render an ascii string to unicode by replacing e.g. greek letters"""
        if len(ascii_str) == 0:
            return ''
        name, supers, subs = split_super_sub(ascii_str)
        name = _translate_symbols(name)
        supers = [_translate_symbols(sup) for sup in supers]
        subs = [_translate_symbols(sub) for sub in subs]
        try:
            supers_modified = [
                    unicode_sub_super(s, superscript_mapping)
                    for s in supers]
            subs_modified = [
                    unicode_sub_super(s, subscript_mapping, 1)
                    for s in subs]
            if len(supers_modified) > 0:
                name += " ".join(supers_modified)
            if len(subs_modified) > 0:
                name += " ".join(subs_modified)
        except KeyError:
            if len(supers) > 0:
                name += "^%s" % " ".join(supers)
            if len(subs) > 0:
                name += "_%s" % " ".join(subs)
        return name

    @classmethod
    def render_op(cls, identifier, hs=None, dagger=False, args=None,
                  superop=False):
        """Render an operator"""
        hs_label = None
        if hs is not None:
            hs_label = cls.render_hs_label(hs)
        name, total_subscript, total_superscript, args_str \
            = cls._split_op(identifier, hs_label, dagger, args)
        if len(name) == 1:
            if superop:
                res = name
            else:
                res = modifier_dict['hat'](name)
        else:
            res = name
        try:
            sub_super = ''
            if len(total_subscript) > 0:
                sub_super += unicode_sub_super(
                             total_subscript, subscript_mapping, 1)
            if len(total_superscript) > 0:
                sub_super += unicode_sub_super(
                             total_superscript, superscript_mapping)
            res += sub_super
        except KeyError:
            if len(total_subscript) > 0:
                res += "_" + total_subscript
            if len(total_superscript) > 0:
                res += "^" + total_superscript
        if len(args_str) > 0:
            res += args_str
        return res

    @classmethod
    def render_scalar(cls, value, adjoint=False):
        """Render a scalar value (numeric or symbolic)"""
        if adjoint:
            value = sympy.conjugate(value)
        res = sympy.pretty(
                value, use_unicode=True, wrap_line=False)
        if "\n" in res:
            res = str(value)
            for string in re.findall(r'[A-Za-z]+', res):
                if string in _greek_dictionary:
                    res = res.replace(string, _greek_dictionary[string])
        return res


_greek_dictionary = {
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


def _translate_symbols(string):
    """Given a description of a Greek letter or other special character,
    return the appropriate latex."""
    res = []
    for s in re.split(r'([,.:\s]+)', string):
        tex_str = _greek_dictionary.get(s)
        if tex_str:
            res.append(tex_str)
        elif s.lower() in _greek_dictionary:
            res.append(_greek_dictionary[s])
        else:
            res.append(s)
    return "".join(res)


def unicode(expr):
    """Return a unicode representation of the given `expr`"""
    return UnicodePrinter.render(expr)


subscript_mapping = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
    '7': '₇', '8': '₈', '9': '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋',
    '=': '₌', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ', 'k': 'ₖ',
    'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
    'β': 'ᵦ', 'γ': 'ᵧ', 'ρ': 'ᵨ', 'φ': 'ᵩ', 'χ': 'ᵪ'
}


superscript_mapping = {
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


def unicode_sub_super(string, mapping, max_len=None):
    """Try to render a subscript string in unicode, fall back on ascii if this
    is not possible"""
    string = str(string)
    if max_len is not None:
        if len(string) > max_len:
            raise KeyError("max_len exceeded")
    unicode_letters = []
    for letter in string:
        unicode_letters.append(mapping[letter])
    return ''.join(unicode_letters)
