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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################
"""Printing system for QNET Expressions and related objects"""

import copy
from contextlib import contextmanager

import sympy
from numpy import int64, complex128, float64

SCALAR_TYPES = (int, float, complex, sympy.Basic, int64, complex128, float64)
# SCALAR_TYPES *must* be defined *before* imports from sub-modules

from .base import Printer
from .ascii import AsciiPrinter, ascii
from .unicode import UnicodePrinter, unicode
from .tex import LaTeXPrinter, tex
from .srepr import SReprPrinter, srepr
from .tree import tree


__all__ = ['init_printing', 'configure_printing', 'ascii', 'unicode', 'tex',
           'srepr', 'tree']


def init_printing(
    use_unicode=True, str_printer=None, repr_printer=None,
    cached_rendering=True, implicit_tensor=False, _init_sympy=True):
    """Initialize printing

    * Initialize `sympy` printing with the given `use_unicode`
      (i.e. call `sympy.init_printing`)
    * Set the printers for textual representations (``str`` and ``repr``) of
      Expressions
    * Configure whether `ascii`, `unicode`, and `tex` representations should be
      cached. If caching is enabled, the representations are rendered only
      once. This means that any configuration of the corresponding printers
      must be made before generating the representation for the first time.

    Args:
        use_unicode (bool): If True, use unicode symbols. If False, restrict to
            ascii. Besides initializing `sympy` printing, this only determins
            the default `str` and `repr` printer. Thus, if `str_printer` and
            `repr_printer` are given, `use_unicode` has almost no effect.
        str_printer (Printer, str, or None): The printer to be used for
            ``str(expr)``. Must be an instance of :class:`Printer` or one of
            the strings 'ascii', 'unicode', 'unicode', 'latex', or 'srepr',
            corresponding to `AsciiPrinter`, `UnicodePrinter`, `LaTeXPrinter`,
            and `SReprPrinter` respectively. If not given, either
            `AsciiPrinter` or `UnicodePrinter` is set, depending on
            `use_unicode`.
        repr_printer (Printer, str, or None): Like `str_printer`, but for
            ``repr(expr)``. This is also what is displayed in an interactive
            Python session
        cached_rendering (bool): Flag whether the results of ``ascii(expr)``,
            ``unicode(expr)``, and ``tex(expr)`` should be cached
        implicit_tensor (bool): If True, don't use tensor product symbols in
            the standard tex representation

    Notes:
        * This routine does not set custom printers for rendering `ascii`,
          `unicode`, and `tex`. To use a non-default printer, you must assign
          directly to the corresponding class attributes of `Expression`.
        * `str` and `repr` representations are never *directly* cached (but the
          printers they delegate to may use caching)

    """
    from qnet.algebra.abstract_algebra import Expression
    sympy.init_printing(use_unicode=use_unicode)
    # Set the default _str_ and _repr_ printers
    printer_codes = {
        'ascii': AsciiPrinter,
        'unicode': UnicodePrinter,
        'tex': LaTeXPrinter,
        'latex': LaTeXPrinter,
        'srepr': SReprPrinter,
    }
    if str_printer is None:
        if use_unicode:
            Expression._str_printer = UnicodePrinter
        else:
            Expression._str_printer = AsciiPrinter
    else:
        if str_printer in printer_codes:
            str_printer = printer_codes[str_printer]
        if not isinstance(str_printer, Printer):
            raise TypeError("str_printer must be a Printer instance")
        Expression._str_printer = str_printer
    if repr_printer is None:
        if use_unicode:
            Expression._repr_printer = UnicodePrinter
        else:
            Expression._repr_printer = AsciiPrinter
    else:
        if repr_printer in printer_codes:
            repr_printer = printer_codes[repr_printer]
        if not isinstance(repr_printer, Printer):
            raise TypeError("repr_printer must be a Printer instance")
        Expression._repr_printer = repr_printer
    if implicit_tensor:
        LaTeXPrinter.tensor_sym = ' '
    # Set cached rendering
    Expression._cached_rendering = cached_rendering


@contextmanager
def configure_printing(**kwargs):
    """context manager for temporarily changing the printing paremters. This
    takes the same values as `init_printing`"""
    from qnet.algebra.abstract_algebra import Expression
    str_printer = Expression._str_printer
    repr_printer = Expression._repr_printer
    cached_rendering = Expression._cached_rendering
    latex_settings = copy.copy(LaTeXPrinter.__dict__)
    init_printing(_init_sympy=False, **kwargs)
    yield
    Expression._str_printer = str_printer
    Expression._repr_printer = repr_printer
    Expression._cached_rendering = cached_rendering
    LaTeXPrinter.__dict__ = latex_settings
