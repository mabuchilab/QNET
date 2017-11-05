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
"""Printing system for QNET Expressions and related objects"""

import sys
from contextlib import contextmanager
from pydoc import locate

from sympy.printing.printer import Printer as SympyPrinter

from .base import QnetBasePrinter
from .asciimod import ascii
from .unicodemod import unicode
from .latexmod import latex
from .sreprmod import srepr
from .treemod import tree
from .dotmod import dotprint


__all__ = ['init_printing', 'configure_printing', 'ascii', 'unicode', 'latex',
           'srepr', 'tree', 'dotprint']


def _printer_cls(label, class_address, require_base=QnetBasePrinter):
    cls = locate(class_address)
    if cls is None:
        raise ValueError("%s '%s' does not exist" % (label, class_address))
    try:
        if require_base is not None:
            if not issubclass(cls, require_base):
                raise ValueError(
                    "%s '%s' must be a subclass of %s"
                    % (label, class_address, require_base.__name__))
    except TypeError:
            raise ValueError(
                "%s '%s' must be a class" % (label, class_address))
    else:
        return cls


PRINT_FUNC = {
    'ascii': ascii,
    'unicode': unicode,
}


def init_printing(
        inifile=None, str_format=None, repr_format=None,
        ascii_printer='qnet.printing.asciimod.QnetAsciiPrinter',
        ascii_sympy_printer='qnet.printing.sympy.SympyStrPrinter',
        unicode_printer='qnet.printing.unicodemod.QnetUnicodePrinter',
        unicode_sympy_printer='qnet.printing.sympy.SympyUnicodePrinter',
        **settings):
    """Initialize printing"""
    # TODO: handle inifile (and recurse)

    # collect settings
    settings_map = {
        'ascii': {},
        'unicode': {}
    }
    global_settings = {}
    for key, val in settings.values():
        is_global = True
        for prefix in settings_map.keys():
            if key.startwith(prefix):
                settings_map[prefix][key[len(prefix):]] = val
                is_global = False
                break
        if is_global:
            global_settings[key] = val
    QnetBasePrinter.set_global_settings(**global_settings)

    # initialize all print functions
    print_cls_map = {
        'ascii': (ascii_printer, ascii_sympy_printer),
        'unicode': (unicode_printer, unicode_sympy_printer),
    }
    for name in print_cls_map.keys():
        print_func = PRINT_FUNC[name]
        qnet_printer_address, sympy_printer_address = print_cls_map[name]
        print_func._printer_cls = _printer_cls(
            name + '_printer', qnet_printer_address)
        print_func._printer_cls.sympy_printer_cls = _printer_cls(
            name + '_sympy_printer', sympy_printer_address,
            require_base=SympyPrinter)
        print_func.printer = print_func._printer_cls(
            settings=settings_map[name])

    # set up the __str__ and __repr__ printers
    has_unicode = "UTF-8" in sys.stdout.encoding
    if str_format is None:
        str_format = 'unicode' if has_unicode else 'ascii'
    try:
        str_func = PRINT_FUNC[str_format]
    except KeyError:
        raise ValueError(
            "str_format must be one of %s" % ", ".join(PRINT_FUNC.keys()))
    if repr_format is None:
        repr_format = 'unicode' if has_unicode else 'ascii'
    try:
        repr_func = PRINT_FUNC[repr_format]
    except KeyError:
        raise ValueError(
            "repr_format must be one of %s" % ", ".join(PRINT_FUNC.keys()))
    from qnet.algebra.abstract_algebra import Expression
    Expression.__str__ = lambda self: str_func(self)
    Expression.__repr__ = lambda self: repr_func(self)
    Expression._repr_latex = lambda self: "$" + latex(self) + "$"


@contextmanager
def configure_printing(**kwargs):
    """context manager for temporarily changing the printing paremters. This
    takes the same values as `init_printing`"""
    # TODO
    #from qnet.algebra.abstract_algebra import Expression
    #str_printer = Expression._str_printer
    #repr_printer = Expression._repr_printer
    #cached_rendering = Expression._cached_rendering
    #latex_settings = copy.copy(LaTeXPrinter.__dict__)
    #init_printing(_init_sympy=False, **kwargs)
    yield
    #Expression._str_printer = str_printer
    #Expression._repr_printer = repr_printer
    #Expression._cached_rendering = cached_rendering
    #LaTeXPrinter.__dict__ = latex_settings
