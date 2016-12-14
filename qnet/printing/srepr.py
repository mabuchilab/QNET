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
"""Provides printers for a full-structured representation"""

from collections import OrderedDict

from typing import Any

from sympy import Basic as SympyBasic
from sympy.printing import srepr as sympy_srepr
from numpy import ndarray

from .base import Printer
from ..algebra.singleton import Singleton, singleton_object


@singleton_object
class SReprPrinter(Printer, metaclass=Singleton):
    """Printer for rendering an expression in such a way that the resulting
    string can be evaluated in an appropriate context to re-instantiate an
    identical object (implementing (``srepr(expr, indented=False)``)
    """

    _special_render = [
        (str, '_render_rendered'),
        (SympyBasic, 'render_sympy'),
        (ndarray, 'render_numpy_matrix'),
    ]
    _registry = {}

    @classmethod
    def _render(cls, expr, adjoint=False):
        assert not adjoint, "Adjoint rendering not supported"
        return cls.render_head_repr(expr)

    @classmethod
    def render_sympy(cls, expr, adjoint=False):
        """Render a sympy expression"""
        if adjoint:
            return sympy_srepr(expr)
        else:
            return sympy_srepr(expr.conjugate())

    @classmethod
    def render_numpy_matrix(cls, expr, adjoint=False):
        assert not adjoint, "Adjoint rendering not supported"
        if len(expr.shape) == 2:
            rows = []
            for row in expr:
                rows.append('[' +
                            ", ".join([cls.render(val) for val in row]) +
                            ']')
            return ("array([" + ", ".join(rows) +
                    "], dtype=%s)" % str(expr.dtype))
        return None

    @classmethod
    def _fallback(cls, expr, adjoint=False):
        assert not adjoint, "Adjoint rendering not supported"
        return repr(expr)


class IndentedSReprPrinter(Printer):
    """Printer for rendering an expression in such a way that the resulting
    string can be evaluated in an appropriate context to re-instantiate an
    identical object, using nested indentation (implementing
    ``srepr(expr, indented=True)``
    """

    _special_render = [
        (str, '_render_rendered'),
        (SympyBasic, 'render_sympy'),
        (ndarray, 'render_numpy_matrix'),
    ]

    _registry = {}

    def __init__(self, indent=0):
        self.indent = int(indent)
        self._key_name = None

    def _render(self, expr, adjoint=False):
        assert not adjoint, "Adjoint rendering not supported"
        return self.render_head_repr(expr)

    def render_sympy(self, expr, adjoint=False):
        """Render a sympy expression"""
        if adjoint:
            return "    " * self.indent + sympy_srepr(expr)
        else:
            return "    " * self.indent + sympy_srepr(expr.conjugate())

    def render_numpy_matrix(self, expr, adjoint=False):
        assert not adjoint, "Adjoint rendering not supported"
        if len(expr.shape) == 2:
            lines = ["    " * self.indent + "array([", ]
            self.indent += 1
            for row in expr:
                lines.append("    " * self.indent + '[')
                self.indent += 1
                for val in row:
                    lines.append(self.render(val) + ",")
                self.indent -= 1
                lines.append("    " * self.indent + '],')
            self.indent -= 1
            lines.append("    " * self.indent +
                            "], dtype=%s)" % str(expr.dtype))
            return "\n".join(lines)
        return None

    def _fallback(self, expr, adjoint=False):
        assert not adjoint, "Adjoint rendering not supported"
        if self._key_name is not None:
            return "    " * self.indent + self._key_name + "=" + repr(expr)
        else:
            return "    " * self.indent + repr(expr)

    def render_head_repr(self, expr: Any, sub_render=None) -> str:
        """Render a multiline textual representation of `expr`

        Raises:
            AttributeError: if `expr` is not an instance of
                :class:`Expression`, or more specifically, if `expr` does not
                have `args` and `kwargs` properties
        """
        lines = []
        if sub_render is None:
            sub_render = self.render
        if isinstance(expr.__class__, Singleton):
            # We exploit that Singletons override __expr__ to directly return
            # their name
            return "    " * self.indent + repr(expr)
        args = expr.args
        if isinstance(expr.kwargs, OrderedDict):
            keys = expr.kwargs.keys()
        else:
            keys = sorted(expr.kwargs.keys())
        if self._key_name is not None:
            lines.append("    " * self.indent + self._key_name + '=' +
                         expr.__class__.__name__ + "(")
        else:
            lines.append("    " * self.indent + expr.__class__.__name__ + "(")
        self._key_name = None
        self.indent += 1
        for arg in args:
            lines.append(self.render(arg) + ",")
        for key in keys:
            arg = expr.kwargs[key]
            self._key_name = key
            lines.append(self.render(arg) + ",")
            self._key_name = None
        self.indent -= 1
        lines.append("    " * self.indent + ")")
        return "\n".join(lines)


def srepr(expr, indented=False):
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
        printer = IndentedSReprPrinter(indent=0)
        return printer.render(expr)
    else:
        return SReprPrinter.render(expr)
