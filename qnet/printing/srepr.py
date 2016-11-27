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
from .base import Printer
from ..algebra.singleton import Singleton, singleton_object

from sympy import Basic as SympyBasic
from sympy.printing import srepr as sympy_srepr
from numpy import ndarray


@singleton_object
class SReprPrinter(Printer, metaclass=Singleton):
    """Printer for rendering an expression in such a way that the resulting
    string can be evaluated in an appropriate context to re-instantiate an
    identical object
    """

    @classmethod
    def render(cls, expr, adjoint=False):
        """Render the given expression. Not that `adjoint` must be False"""
        assert not adjoint, "adjoint not supported for SReprPrinter"
        if isinstance(expr, SympyBasic):
            return sympy_srepr(expr)
        elif isinstance(expr, ndarray):
            if len(expr.shape) == 2:
                rows = []
                for row in expr:
                    rows.append('[' +
                                ", ".join([cls.render(val) for val in row]) +
                                ']')
                return ("array([" + ", ".join(rows) +
                        "], dtype=%s)" % str(expr.dtype))
        try:
            return cls.render_head_repr(expr)
        except AttributeError:
            return repr(expr)


def srepr(expr):
    """Render the given expression into a string that can be evaluated in an
    appropriate context to re-instantiate an identical expression
    """
    return SReprPrinter.render(expr)
