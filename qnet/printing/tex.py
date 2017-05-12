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
from sympy.printing.conventions import split_super_sub
from sympy.printing import latex as sympy_latex
from sympy.printing.latex import (greek_letters_set, other_symbols)
from numpy import complex128

from .base import Printer
from ..algebra.singleton import Singleton, singleton_object
from ..algebra.scalar_types import SCALAR_TYPES


@singleton_object
class LaTeXPrinter(Printer, metaclass=Singleton):
    """Printer for LaTeX. See :class:`Printer` for overview of class
    attributes and other details.
    """

    head_repr_fmt = r'{{\rm {head}\left({args}{kwargs}\right)}}'
    identity_sym = r'\mathbb{1}'
    circuit_identity_fmt = r'{{\rm cid}}({cdim})'
    zero_sym = r'\mathbb{0}'
    dagger_sym = r'\dagger'
    daggered_sym = r'^{\dagger}'
    permutation_sym = r'\mathbf{P}_{\sigma}'
    pseudo_daggered_sym = r'^{+}'
    par_left = r'\left('
    par_right = r'\right)'
    brak_left = r'\left['
    brak_right = r'\right]'
    arg_sep = ", "
    scalar_product_sym = r''
    tensor_sym = r'\otimes'
    inner_product_sym = r'\cdot'
    op_product_sym = r' '
    circuit_series_sym = r'\lhd'
    circuit_concat_sym = r'\boxplus'
    circuit_inverse_fmt = r'\left[{operand}\right]^{{\rhd}}'
    circuit_fb_fmt = (r'\left\lfloor{{{operand}}}\right\rfloor_'
                      r'{{{output}\rightarrow{{}}{input}}}')
    op_trace_fmt = r'{{\rm tr}}_{{{space}}}\left[{operand}\right]'
    null_space_proj_sym = "P_{Ker}"
    hilbert_space_fmt = r'\mathcal{{H}}_{{{label}}}'
    matrix_left_sym = r'\begin{pmatrix}'
    matrix_right_sym = r'\end{pmatrix}'
    matrix_row_left_sym = r''
    matrix_row_right_sym = r''
    matrix_col_sep_sym = r' & '
    matrix_row_sep_sym = r' \\'
    bra_fmt = r'\left\langle{{}}{label}\right|_{{({space})}}'
    ket_fmt = r'\left|{label}\right\rangle_{{({space})}}'
    ketbra_fmt = (r'\left|{label_i}\right\rangle\left\langle{{}}{label_j}'
                  r'\right|_{{({space})}}')
    braket_fmt = (r'\left\langle{{}}{label_i}\right|\left.{label_j}'
                  r'\right\rangle_{{({space})}}')
    commut_fmt = r'\left[{A}, {B}\right]'
    anti_commut_fmt = r'\left\{{{A}, {B}\right\}}'
    cc_string = r'\text{c.c.}'

    op_accent = r'\hat'
    superop_accent = r'\mathrm'

    _special_render = [
        (SCALAR_TYPES, 'render_scalar'),
        (str, '_render_rendered'),
        ((sympy.Basic, sympy.Matrix), 'render_sympy'),
    ]
    _registry = {}

    @classmethod
    def _render(cls, expr, adjoint=False):
        return expr._tex_(adjoint=adjoint)

    @classmethod
    def _fallback(cls, expr, adjoint=False):
        """Render an expression that does not have _delegate_mtd"""
        if adjoint:
            return r"{\rm Adjoint[%s]}" % str(expr)
        else:
            return r"{\rm " + str(expr) + "}"

    @classmethod
    def render_sympy(cls, expr, adjoint=False):
        """Render a sympy expression"""
        if adjoint:
            return sympy_latex(expr.conjugate()).strip('$')
        else:
            return sympy_latex(expr).strip('$')

    @classmethod
    def render_hs_label(cls, hs: Any) -> str:
        """Render the total label for the given Hilbert space"""
        # This differs from the base `render_hs_label` only in the use of
        # padding
        if isinstance(hs.__class__, Singleton):
            return cls.render_string(hs.label)
        else:
            return cls.render_product(
                    [cls.render_string(ls.label) for ls in hs.local_factors],
                    prod_sym=cls.tensor_sym, sum_classes=(), padding=' ')

    @classmethod
    def render_op(cls, identifier, hs=None, dagger=False, args=None,
                  superop=False):
        """Render an operator"""
        hs_label = None
        if hs is not None:
            hs_label = cls.render_hs_label(hs)
        name, total_subscript, total_superscript, args_str \
            = cls._split_op(identifier, hs_label, dagger, args)
        op_accent = cls.op_accent
        if superop:
            op_accent = cls.superop_accent
        if not name.startswith(r'\text{'):
            res = op_accent + '{' + name + '}'
        else:
            res = name
        if len(total_subscript) > 0:
            res += "_{" + total_subscript + '}'
        if len(total_superscript) > 0:
            res += "^{" + total_superscript + '}'
        if len(args_str) > 0:
            res += args_str
        return res

    @classmethod
    def render_string(cls, ascii_str):
        """Render an unrendered (ascii) string, resolving e.g. greek letters
        and sub-/superscripts"""
        if len(ascii_str) == 0:
            return ''
        name, supers, subs = split_super_sub(ascii_str)
        name = _translate_symbols(name)
        supers = [_translate_symbols(sup) for sup in supers]
        subs = [_translate_symbols(sub) for sub in subs]
        if len(supers) > 0:
            name += "^{%s}" % " ".join(supers)
        if len(subs) > 0:
            name += "_{%s}" % " ".join(subs)
        return name

    @classmethod
    def render_scalar(cls, value, adjoint=False):
        """Render a scalar value (numeric or symbolic)"""
        if adjoint:
            res = sympy.latex(sympy.conjugate(value))
        else:
            res = sympy.latex(value)
        if isinstance(value, (complex, complex128)):
            if value.real != 0 and value.imag != 0:
                res = cls.par_left + res + cls.par_right
        return res


def tex(expr):
    """Return a LaTeX string representation of the given `expr`"""
    return LaTeXPrinter.render(expr)


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
