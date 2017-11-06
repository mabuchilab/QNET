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
"""ASCII Printer"""
from .base import QnetBasePrinter
from .sympy import SympyStrPrinter


class QnetAsciiPrinter(QnetBasePrinter):
    """Printer for a string (ASCII) representation."""
    sympy_printer_cls = SympyStrPrinter
    printmethod = '_ascii'


def ascii(expr, cache=None, **settings):
    """Return an ascii textual representation of the given object /
    expression"""
    try:
        if cache is None and len(settings) == 0:
            return ascii.printer.doprint(expr)
        else:
            printer = ascii._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        ascii._printer_cls = QnetAsciiPrinter
        ascii.printer = ascii._printer_cls()
        return ascii(expr, cache, **settings)
