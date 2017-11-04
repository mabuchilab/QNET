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
"""Unicode Printer"""
from .base import QnetBasePrinter
from .sympy import SympyUnicodePrinter


class QnetUnicodePrinter(QnetBasePrinter):
    """Printer for a string (Unicode) representation."""
    sympy_printer_cls = SympyUnicodePrinter
    printmethod = '_unicode'


def unicode(expr, **options):
    """Return a unicode textual representation of the given object /
    expression"""
    try:
        if len(options) == 0:
            return unicode.printer.doprint(expr)
        else:
            return unicode._printer_cls(**options).doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        unicode._printer_cls = QnetUnicodePrinter
        unicode.printer = unicode._printer_cls()
        return unicode(expr, **options)
