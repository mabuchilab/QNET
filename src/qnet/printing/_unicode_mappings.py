"""Mappings between ASCII and unicode"""
import re

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
    'vartheta': r'θ', 'up': r'↑', 'down': r'↓', 'uparrow': r'↑',
    'downarrow': r'↓',
}

_SUBSCRIPT_MAPPING = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
    '7': '₇', '8': '₈', '9': '₉', '(': '₍', ')': '₎', '+': '₊', '-': '₋',
    '=': '₌', 'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ', 'h': 'ₕ', 'k': 'ₖ',
    'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
    'β': 'ᵦ', 'γ': 'ᵧ', 'ρ': 'ᵨ', 'φ': 'ᵩ', 'χ': 'ᵪ', ' ': ' ',
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
    'θ': 'ᶿ', 'ι': 'ᶥ', 'φ': 'ᵠ', 'χ': 'ᵡ', ' ': ' ',
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
    if string.startswith('(') and string.endswith(')'):
        len_string = len(string) - 2
    else:
        len_string = len(string)
    if max_len is not None:
        if len_string > max_len:
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
