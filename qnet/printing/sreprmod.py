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
"""Provides printers for a full-structured representation"""

from sympy.printing.repr import (
    ReprPrinter as SympyReprPrinter, srepr as sympy_srepr)
from sympy.core.basic import Basic as SympyBasic

from .base import QnetBasePrinter
from ..algebra.abstract_algebra import Expression
from ..algebra.singleton import Singleton


class QnetSReprPrinter(QnetBasePrinter):
    """Printer for a string (ASCII) representation."""

    sympy_printer_cls = SympyReprPrinter

    def _print_ndarray(self, expr):
        if len(expr.shape) == 2:
            rows = []
            for row in expr:
                rows.append(
                    '[' + ", ".join([self.doprint(val) for val in row]) + ']')
            return ("array([" + ", ".join(rows) +
                    "], dtype=%s)" % str(expr.dtype))
        else:
            raise ValueError("Cannot render %s" % expr)


class IndentedSympyReprPrinter(SympyReprPrinter):
    """Indented repr printer for Sympy objects"""

    def doprint(self, expr):
        res = super().doprint(expr)
        return "    " * (self._print_level - 1) + res


class IndentedSReprPrinter(QnetBasePrinter):
    """Printer for rendering an expression in such a way that the resulting
    string can be evaluated in an appropriate context to re-instantiate an
    identical object, using nested indentation (implementing
    ``srepr(expr, indented=True)``
    """

    sympy_printer_cls = IndentedSympyReprPrinter

    def __init__(self, cache=None, settings=None):
        self._key_name = None
        super().__init__(cache=cache, settings=settings)

    def emptyPrinter(self, expr):
        """Fallback printer"""
        indent = "    " * (self._print_level - 1)
        lines = []
        if isinstance(expr.__class__, Singleton):
            # We exploit that Singletons override __expr__ to directly return
            # their name
            return indent + repr(expr)
        if isinstance(expr, Expression):
            args = expr.args
            keys = expr.minimal_kwargs.keys()
            if self._key_name is not None:
                lines.append(
                    indent + self._key_name + '=' + expr.__class__.__name__ +
                    "(")
            else:
                lines.append(indent + expr.__class__.__name__ + "(")
            self._key_name = None
            for arg in args:
                lines.append(self.doprint(arg) + ",")
            for key in keys:
                arg = expr.kwargs[key]
                self._key_name = key
                lines.append(self.doprint(arg) + ",")
                self._key_name = None
        elif isinstance(expr, SympyBasic):
            lines.append(indent + sympy_srepr(expr))
        else:
            lines.append(indent + repr(expr))
        lines.append(indent + ")")
        return "\n".join(lines)

    def _print_ndarray(self, expr):
        indent = "    " * (self._print_level - 1)
        if len(expr.shape) == 2:
            lines = [+ "array([", ]
            for row in expr:
                lines.append(indent + '[')
                for val in row:
                    lines.append(self.doprint(val) + ",")
                lines.append(indent + '],')
            lines.append(indent + "], dtype=%s)" % str(expr.dtype))
            return "\n".join(lines)
        else:
            raise ValueError("Cannot render %s" % expr)


def srepr(expr, indented=False, cache=None):
    """Render the given expression into a string that can be evaluated in an
    appropriate context to re-instantiate an identical expression. If
    `indented` is False (default), the resulting string is a single line.
    Otherwise, the result is a multiline string, and each positional and
    keyword argument of each `Expression` is on a separate line, recursively
    indented to produce a tree-like output.

    See also:
        `qnet.printing.tree_str` produces an output similar to `srepr` with
        ``indented=True``. Unlike `srepr`, however, `tree_str` uses line
        drawings for the tree, shows arguments directly on the same line as the
        expression they belong to, and cannot be evaluated.
    """
    if indented:
        printer = IndentedSReprPrinter(cache=cache)
    else:
        printer = QnetSReprPrinter(cache=cache)
    return printer.doprint(expr)
