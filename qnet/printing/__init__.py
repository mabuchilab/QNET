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
from ._ascii import QnetAsciiPrinter
from ._unicode import QnetUnicodePrinter
from ._latex import QnetLatexPrinter
from ._srepr import QnetSReprPrinter, IndentedSReprPrinter
from .tree import tree_str as _tree_str

# import submodules for quick interactive access
import qnet.printing.tree
import qnet.printing.dot

__all__ = ['init_printing', 'configure_printing', 'ascii', 'unicode', 'latex',
           'srepr']


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


def init_printing(reset=False, **kwargs):
    """Initialize the printing system.

    This determines the behavior of the :func:`ascii`, :func:`unicode`,
    and :func:`latex` functions, as well as the ``__str__`` and ``__repr__`` of
    any QNET Expression.

    The routine may be called in one of two forms. First,

    ::

        init_printing(inifile=<path_to_file>)

    Second,

    ::

        init_printing(str_format=<str_fmt>, repr_format=<repr_fmt>,
                      caching=<use_caching>, **settings)

    provides a simplified, "manual" setup with the parameters below.

    Args:
        str_format (str): Format for ``_str_``
        repr_format (str): Format for ``__repr``
        caching (bool): Whether to allow caching
        settings: Any setting understood

    Generally, this function should be called only once at the beginning of a
    script or notebook.
    """
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
        ascii_printer='qnet.printing._ascii.QnetAsciiPrinter',
        ascii_sympy_printer='qnet.printing.sympy.SympyStrPrinter',
        unicode_printer='qnet.printing._unicode.QnetUnicodePrinter',
        unicode_sympy_printer='qnet.printing.sympy.SympyUnicodePrinter',
        latex_printer='qnet.printing._latex.QnetLatexPrinter',
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

        print_func = _PRINT_FUNC[name]
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
    _PRINT_FUNC['tree'] = partial(_tree_str, unicode=has_unicode)
    if str_format is None:
        str_format = 'unicode' if has_unicode else 'ascii'
        logger.debug("Setting __str__ format to %s", str_format)
    try:
        str_func = _PRINT_FUNC[str_format]
    except KeyError:
        raise ValueError(
            "str_format must be one of %s" % ", ".join(_PRINT_FUNC.keys()))
    if repr_format is None:
        repr_format = 'unicode' if has_unicode else 'ascii'
        logger.debug("Setting __repr__ format to %s" % repr_format)
    try:
        repr_func = _PRINT_FUNC[repr_format]
    except KeyError:
        raise ValueError(
            "repr_format must be one of %s" % ", ".join(_PRINT_FUNC.keys()))
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


def unicode(expr, cache=None, **settings):
    """Return a unicode textual representation of the given object /
    expression"""
    try:
        if cache is None and len(settings) == 0:
            return unicode.printer.doprint(expr)
        else:
            printer = unicode._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        unicode._printer_cls = QnetUnicodePrinter
        unicode.printer = unicode._printer_cls()
        return unicode(expr, cache, **settings)


def latex(expr, cache=None, **settings):
    """Return a LaTeX textual representation of the given object /
    expression"""
    try:
        if cache is None and len(settings) == 0:
            return latex.printer.doprint(expr)
        else:
            printer = latex._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        latex._printer_cls = QnetLatexPrinter
        latex.printer = latex._printer_cls()
        return latex(expr, cache, **settings)


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


# Map acceptable values for `str_format` and `repr`_format in
# `init_printing` to a print function
_PRINT_FUNC = {
    'ascii': ascii,
    'unicode': unicode,
    'latex': latex,
    'tex': latex,
    'srepr': srepr,
    'tree': _tree_str,   # init_printing will modify this for unicode support
}
