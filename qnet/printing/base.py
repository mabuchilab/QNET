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
"""Provides the base class for Printers"""

from typing import Any

from sympy.core.basic import Basic as SympyBasic
from sympy.printing.printer import Printer as SympyPrinter
from sympy.printing.repr import srepr as sympy_srepr

from ..algebra.singleton import Singleton
from ..algebra.abstract_algebra import Expression
from .sympy import SympyStrPrinter


class QnetBasePrinter(SympyPrinter):
    """Base class for all QNET expression printers

    Args:
        cache (dict or None): A dict that maps expressions to strings. It may
            be given during istantiation to use pre-defined strings for
            specific expressions. The cache will be updated as the printer is
            used.
        settings (dict or None): A dict of settings. All settings that start
            with the prefix ``sympy_`` are used for the
            :attr:`sympy_printer_cls` (without the prefix). The remaining
            settings must have keys that are in :attr:`_default_settings`.

    Class Attributes:
        sympy_printer_cls (type): The class that will be instantiated to print
            Sympy expressions
        _default_settings (dict): The default value of all settings. Note only
            settings for which there are defaults defined here are accepted
            when instantiating the printer
        printmethod (None or str): Name of a method that expressions may define
            to print themeselves.

    Attributes:
        cache (dict): Dictionary where the results of any call to
            :meth:`doprint` is stored. When :meth:`doprint` is called for an
            expression that is already in :attr:`cache`, the result from the
            cache is returned.
        _sympy_printer (sympy.printing.printer.Printer): The printer instance
            that will be used to print any Sympy expression.
        _print_level (int): The recursion depth of :meth:`doprint`
            (>= 1 inside any of the ``_print*`` methods)

    Raises:
        TypeError: If any key in `settings` is not defined in the
            :attr:`_default_settings` of the printer, respectively the
            :attr:`sympy_printer_cls`.
    """

    sympy_printer_cls = SympyStrPrinter

    _default_settings = {
        'show_hilbert_space': True,
        'head_repr_fmt': r'{head}({args}{kwargs})'
    }

    printmethod = None

    def __init__(self, cache=None, settings=None):
        self.cache = {}
        if cache is not None:
            self.cache = cache
        sympy_settings = {}
        qnet_settings = {}
        if settings is not None:
            for key, val in settings.items():
                if key.startswith('sympy_'):
                    sympy_settings[key[6:]] = val
                else:
                    qnet_settings[key] = val
        self._sympy_printer = self.sympy_printer_cls(settings=sympy_settings)
        super().__init__(settings=qnet_settings)

    def emptyPrinter(self, expr):
        """Fallback method for expressions that neither know how to print
        themeselves, nor for which the printer has a suitable ``_print*``
        method"""
        return render_head_repr(expr)

    def doprint(self, expr):
        """Returns printer's representation for expr (as a string)

        The representation is obtained by the following methods:

        1. from the :attr:`cache`
        2. If `expr` is a Sympy object, delegate to the
           :meth:`~sympy.printing.printe.Printer.doprint` method of
           :attr:`_sympy_printer`
        3. Let the `expr` print itself if has the :attr:`printmethod`
        4. Take the best fitting ``_print_*`` method of the printer
        5. As fallback, delegate to :meth:`emptyPrinter`
        """
        if expr in self.cache:
            return self.cache[expr]
        else:
            if isinstance(expr, SympyBasic):
                res = self._sympy_printer.doprint(expr)
            else:
                # the _print method, inherited from SympyPrinter implements the
                # internal dispatcher for (3-5)
                res = self._str(self._print(expr))
        self.cache[expr] = res
        return res


def render_head_repr(
        expr: Any, sub_render=None, key_sub_render=None) -> str:
    """Render a textual representation of `expr` using
    Positional and keyword arguments are recursively
    rendered using `sub_render`, which defaults to `render_head_repr` by
    default.  If desired, a different renderer may be used for keyword
    arguments by giving `key_sub_renderer`

    Raises:
        AttributeError: if `expr` is not an instance of
            :class:`Expression`, or more specifically, if `expr` does not
            have `args` and `kwargs` (respectively `minimal_kwargs`)
            properties
    """
    head_repr_fmt = r'{head}({args}{kwargs})'
    if sub_render is None:
        sub_render = render_head_repr
    if key_sub_render is None:
        key_sub_render = sub_render
    if isinstance(expr.__class__, Singleton):
        # We exploit that Singletons override __expr__ to directly return
        # their name
        return repr(expr)
    if isinstance(expr, Expression):
        args = expr.args
        keys = expr.minimal_kwargs.keys()
        kwargs = ''
        if len(keys) > 0:
            kwargs = ",".join(
                        ["%s=%s" % (key, key_sub_render(expr.kwargs[key]))
                            for key in keys])
            if len(args) > 0:
                kwargs = "," + kwargs
        return head_repr_fmt.format(
            head=expr.__class__.__name__,
            args=",".join([sub_render(arg) for arg in args]),
            kwargs=kwargs)
    elif isinstance(expr, SympyBasic):
        return sympy_srepr(expr)
    else:
        return repr(expr)
