# This file is part of QNET.
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

import pytest

from sympy import symbols, sqrt, exp, I

from qnet.algebra.circuit_algebra import CircuitSymbol, Feedback
from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import OperatorSymbol
from qnet.printing import (
    configure_printing, SReprPrinter, AsciiPrinter, UnicodePrinter, ascii, tex)


def test_custom_str_repr_printer():
    """Test the ascii representation of "atomic" circuit algebra elements"""
    expr = CircuitSymbol("Xi_2", 2)
    assert str(expr) == 'Ξ₂'
    assert repr(expr) == 'Ξ₂'
    with configure_printing(use_unicode=False):
        assert str(expr) == 'Xi_2'
        assert repr(expr) == 'Xi_2'
    assert str(expr) == 'Ξ₂'
    assert repr(expr) == 'Ξ₂'
    with configure_printing(str_printer='ascii', repr_printer='srepr'):
        assert str(expr) == 'Xi_2'
        assert repr(expr) == "CircuitSymbol('Xi_2', 2)"
    assert str(expr) == 'Ξ₂'
    assert repr(expr) == 'Ξ₂'
    with configure_printing(use_unicode=False, repr_printer=SReprPrinter):
        assert str(expr) == 'Xi_2'
        assert repr(expr) == "CircuitSymbol('Xi_2', 2)"
    with configure_printing(str_printer=AsciiPrinter,
                            repr_printer=UnicodePrinter):
        assert str(expr) == 'Xi_2'
        assert repr(expr) == 'Ξ₂'
    with configure_printing(str_printer='tex', repr_printer='ascii'):
        assert str(expr) == r'\Xi_{2}'
        assert repr(expr) == 'Xi_2'


def test_no_cached_rendering():
    """Test that we can temporarily suspend caching"""
    expr = Feedback(CircuitSymbol("Xi_2", 2), out_port=1, in_port=0)
    assert ascii(expr) == '[Xi_2]_{1->0}'
    orig_circuit_fb_fmt = AsciiPrinter.circuit_fb_fmt
    AsciiPrinter.circuit_fb_fmt = r'FB[{operand}]'
    assert ascii(expr) == '[Xi_2]_{1->0}'  # cached
    with configure_printing(cached_rendering=False):
        assert ascii(expr) == 'FB[Xi_2]'
    assert ascii(expr) == '[Xi_2]_{1->0}'  # cached
    AsciiPrinter.circuit_fb_fmt = orig_circuit_fb_fmt


def test_implicit_tensor():
    """Test the implicit_tensor printing options"""
    A = OperatorSymbol('A', hs=1)
    B = OperatorSymbol('B', hs=2)
    assert tex(A*B) == r'\hat{A}^{(1)} \otimes \hat{B}^{(2)}'
    with configure_printing(implicit_tensor=True, cached_rendering=False):
        assert tex(A*B) == r'\hat{A}^{(1)} \hat{B}^{(2)}'
