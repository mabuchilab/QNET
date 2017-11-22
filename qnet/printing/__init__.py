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
import logging
from contextlib import contextmanager
from collections import defaultdict
from functools import partial
from pydoc import locate

from sympy.printing.printer import Printer as SympyPrinter

from .base import QnetBasePrinter
from .asciimod import ascii
from .unicodemod import unicode
from .latexmod import latex
from .sreprmod import srepr
from .treemod import tree, tree_str as _tree_str
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


# Map acceptable values for `str_format` and `repr`_format in
# `init_printing` to a print function
PRINT_FUNC = {
    'ascii': ascii,
    'unicode': unicode,
    'latex': latex,
    'tex': latex,
    'srepr': srepr,
    'tree': _tree_str,   # init_printing will modify this for unicode support
}


def init_printing(reset=False, **kwargs):
    """Initialize printing"""
    if reset:
        SympyPrinter._global_settings = {}
    if 'inifile' in kwargs:
        assert len(kwargs) == 1
        raise NotImplementedError("INI file initalization not implemented yet")
    else:
        # return either None (default) or a dict of frozen attributes if
        # ``_freeze=True`` is given as a keyword argument (internal use in
        # `configure_printing` only)
        return _init_printing(**kwargs)


def _init_printing(
        str_format=None, repr_format=None, caching=True,
        ascii_printer='qnet.printing.asciimod.QnetAsciiPrinter',
        ascii_sympy_printer='qnet.printing.sympy.SympyStrPrinter',
        unicode_printer='qnet.printing.unicodemod.QnetUnicodePrinter',
        unicode_sympy_printer='qnet.printing.sympy.SympyUnicodePrinter',
        latex_printer='qnet.printing.latexmod.QnetLatexPrinter',
        latex_sympy_printer='qnet.printing.sympy.SympyLatexPrinter',
        _freeze=False, **settings):
    logger = logging.getLogger(__name__)
    freeze = defaultdict(dict)
    freeze[SympyPrinter]['_global_settings'] \
        = SympyPrinter._global_settings.copy()
    # Putting the settings in the _global_settings dict for SympyPrinter makes
    # sure that any setting that is acceptable to any Printer that is newly
    # instantiated is used automatically. Settings in _global_settings that are
    # not in the _default_settings of a Printer will be silently ignored.
    SympyPrinter.set_global_settings(**settings)
    # Note that this is the *only* mechanism by which we handle the settings;
    # no settings are passed to any specific Printer below -- Printer-specific
    # settings are possible only when using an INI file.

    freeze[QnetBasePrinter]['_allow_caching'] = QnetBasePrinter._allow_caching
    QnetBasePrinter._allow_caching = caching

    print_cls_map = {
        # print fct     printer cls     sympy printer class
        'ascii':   (ascii_printer,   ascii_sympy_printer),
        'unicode': (unicode_printer, unicode_sympy_printer),
        'latex':   (latex_printer,   latex_sympy_printer),
    }

    for name in print_cls_map.keys():

        print_func = PRINT_FUNC[name]
        qnet_printer_address, sympy_printer_address = print_cls_map[name]

        if hasattr(print_func, '_printer_cls'):
            freeze[print_func]['_printer_cls'] = print_func._printer_cls
            freeze[print_func._printer_cls]['sympy_printer_cls'] = \
                print_func._printer_cls.sympy_printer_cls
        if hasattr(print_func, 'printer'):
            freeze[print_func]['printer'] = print_func.printer

        print_func._printer_cls = _printer_cls(
            name + '_printer', qnet_printer_address)
        print_func._printer_cls.sympy_printer_cls = _printer_cls(
            name + '_sympy_printer', sympy_printer_address,
            require_base=SympyPrinter)
        # instantiation of sympy_printer happens in init-routine (which is why
        # the sympy_printer_cls must be set first!)
        print_func.printer = print_func._printer_cls()

    # set up the __str__ and __repr__ printers
    try:
        has_unicode = "UTF-8" in sys.stdout.encoding
    except TypeError:
        has_unicode = False
    logger.debug(
        "Terminal supports unicode: %s (autodetect)", has_unicode)
    PRINT_FUNC['tree'] = partial(_tree_str, unicode=has_unicode)
    if str_format is None:
        str_format = 'unicode' if has_unicode else 'ascii'
        logger.debug("Setting __str__ format to %s", str_format)
    try:
        str_func = PRINT_FUNC[str_format]
    except KeyError:
        raise ValueError(
            "str_format must be one of %s" % ", ".join(PRINT_FUNC.keys()))
    if repr_format is None:
        repr_format = 'unicode' if has_unicode else 'ascii'
        logger.debug("Setting __repr__ format to %s" % repr_format)
    try:
        repr_func = PRINT_FUNC[repr_format]
    except KeyError:
        raise ValueError(
            "repr_format must be one of %s" % ", ".join(PRINT_FUNC.keys()))
    from qnet.algebra.abstract_algebra import Expression
    freeze[Expression]['__str__'] = Expression.__str__
    freeze[Expression]['__repr__'] = Expression.__repr__
    freeze[Expression]['_repr_latex_'] = Expression._repr_latex_
    Expression.__str__ = lambda self: str_func(self)
    Expression.__repr__ = lambda self: repr_func(self)
    Expression._repr_latex = lambda self: "$" + latex(self) + "$"
    if _freeze:
        return freeze


@contextmanager
def configure_printing(**kwargs):
    """context manager for temporarily changing the printing paremters. This
    takes the same values as `init_printing`"""
    freeze = init_printing(_freeze=True, **kwargs)
    yield
    for obj, attr_map in freeze.items():
        for attr, val in attr_map.items():
            setattr(obj, attr, val)
