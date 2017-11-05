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
#
###########################################################################
"""
Routines for rendering expressions to LaTeX
"""
import re
from typing import Any

import sympy
from sympy.printing.latex import (greek_letters_set, other_symbols)

from .base import QnetBasePrinter
from .sympy import SympyLatexPrinter


class QnetLatexPrinter(QnetBasePrinter):
    """Printer for a LaTeX representation."""
    sympy_printer_cls = SympyLatexPrinter
    printmethod = '_latex'


def latex(expr, cache=None, **settings):
    """Return a LaTeX textual representation of the given object /
    expression"""
    try:
        if cache is None and len(settings) == 0:
            return latex.printer.doprint(expr)
        else:
            return latex._printer_cls(cache, settings).doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        latex._printer_cls = QnetLatexPrinter
        latex.printer = latex._printer_cls()
        return latex(expr, cache, **settings)


_tex_greek_dictionary = {
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
    'varsigma': r'\varsigma', 'vartheta': r'\vartheta',
}


def _translate_symbols(string):
    """Given a description of a Greek letter or other special character,
    return the appropriate latex."""
    res = []
    for s in re.split(r'([,.:\s]+)', string):
        tex_str = _tex_greek_dictionary.get(s)
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
