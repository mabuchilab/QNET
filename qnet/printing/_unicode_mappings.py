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
"""Mappings between ascii and unicode"""

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


subscript_mapping = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
    '7': '₇', '8': '₈', '9': '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋',
    '=': '₌', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ', 'k': 'ₖ',
    'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
    'β': 'ᵦ', 'γ': 'ᵧ', 'ρ': 'ᵨ', 'φ': 'ᵩ', 'χ': 'ᵪ', ' ': ' ',
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
    'θ': 'ᶿ', 'ι': 'ᶥ', 'φ': 'ᵠ', 'χ': 'ᵡ', ' ': ' ',
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
