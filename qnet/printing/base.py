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

from sympy.core.basic import Basic as SympyBasic
from sympy.printing.printer import Printer as SympyPrinter

from ..algebra.scalar_types import SCALAR_TYPES
from .sympy import SympyStrPrinter
from ._render_head_repr import render_head_repr

__all__ = ['QnetBasePrinter']


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
        _allow_caching (bool): A flag that may be set to completely disable
            caching
        _print_level (int): The recursion depth of :meth:`doprint`
            (>= 1 inside any of the ``_print*`` methods)

    Raises:
        TypeError: If any key in `settings` is not defined in the
            :attr:`_default_settings` of the printer, respectively the
            :attr:`sympy_printer_cls`.
    """

    sympy_printer_cls = SympyStrPrinter

    _default_settings = {
        'show_hilbert_space': True,  # alternatively: False, 'subscript'
        'local_sigma_as_ketbra': True,
    }

    _allow_caching = True  # DEBUG

    printmethod = None

    def __init__(self, cache=None, settings=None):
        # This would be more elegant if we declared this as **settings, but we
        # want to closely mirror the SymPy implementation. The frontend-facing
        # routines, e.g. init_printing or ascii do flatten out the settings
        self.cache = {}
        if cache is not None:
            self.cache = cache
        sympy_settings = {}
        qnet_settings = {}
        # TODO: just pass settings to sympy if they're acceptable
        if settings is not None:
            for key, val in settings.items():
                if key.startswith('sympy_'):
                    sympy_settings[key[6:]] = val
                else:
                    qnet_settings[key] = val
        self._sympy_printer = self.sympy_printer_cls(settings=sympy_settings)
        super().__init__(settings=qnet_settings)

    def _render_str(self, string):
        return str(string)

    def emptyPrinter(self, expr):
        """Fallback method for expressions that neither know how to print
        themeselves, nor for which the printer has a suitable ``_print*``
        method"""
        return render_head_repr(expr)

    @staticmethod
    def _isinstance(expr, classname):
        """Check whether `expr` is an instance of the class with name
        `classname`

        This is like the builtin `isinstance`, but it take the `classname` a
        string, instead of the class directly. Useful for when we don't want to
        import the class for which we want to check (also, remember that
        printer choose rendering method based on the class name, so this is
        totally ok)
        """
        for cls in type(expr).__mro__:
            if cls.__name__ == classname:
                return True
        return False

    def _get_from_cache(self, expr):
        """Get the result of :meth:`doprint` from the internal cache"""
        # The reason method this is separated out from `doprint` is that
        # printers that use identation, e.g. IndentedSReprPrinter, need to
        # override how caching is handled, applying variable indentation even
        # for cached results
        try:
            is_cached = expr in self.cache
        except TypeError:
            # expr is unhashable
            is_cached = False
        if is_cached:
            return True, self.cache[expr]
        else:
            return False,  None

    def _write_to_cache(self, expr, res):
        """Store the result of :meth:`doprint` in the internal cache"""
        # Well be overwritten by printers that use indentation, see
        # _get_from_cache
        try:
            self.cache[expr] = res
        except TypeError:
            # expr is unhashable
            pass

    def _print_SCALAR_TYPES(self, expr, *args, **kwargs):
        """Render scalars"""
        adjoint = kwargs.get('adjoint', False)
        if adjoint:
            expr = expr.conjugate()
        if isinstance(expr, SympyBasic):
            self._sympy_printer._print_level = self._print_level + 1
            res = self._sympy_printer.doprint(expr)
        else:
            try:
                if int(expr) == expr:
                    # In Python, objects that evaluate equal (e.g. 2.0 == 2)
                    # have the same hash. We want to normalize this, so that we
                    # get consistent results when printing with a cache
                    expr = int(expr)
            except TypeError:
                pass
            if adjoint:
                kwargs = {
                    key: val for (key, val) in kwargs.items()
                    if key != 'adjoint'}
            res = self._render_str(self._print(expr, *args, **kwargs))
        return res

    def doprint(self, expr, *args, **kwargs):
        """Returns printer's representation for expr (as a string)

        The representation is obtained by the following methods:

        1. from the :attr:`cache`
        2. If `expr` is a Sympy object, delegate to the
           :meth:`~sympy.printing.printe.Printer.doprint` method of
           :attr:`_sympy_printer`
        3. Let the `expr` print itself if has the :attr:`printmethod`
        4. Take the best fitting ``_print_*`` method of the printer
        5. As fallback, delegate to :meth:`emptyPrinter`

        Any extra `args` or `kwargs` are passed to the internal `_print`
        method.
        """
        allow_caching = self._allow_caching
        is_cached = False
        if len(args) > 0 or len(kwargs) > 0:
            # we don't want to cache "custom" rendering, such as the adjoint of
            # the actual expression (kwargs['adjoint'] is True). Otherwise, we
            # might return a cached values for args/kwargs that are different
            # from the the expression was originally cached.
            allow_caching = False

        if allow_caching:
            is_cached, res = self._get_from_cache(expr)
        if not is_cached:
            if isinstance(expr, SCALAR_TYPES):
                res = self._print_SCALAR_TYPES(expr, *args, **kwargs)
            else:
                # the _print method, inherited from SympyPrinter implements the
                # internal dispatcher for (3-5)
                res = self._str(self._print(expr, *args, **kwargs))
            if allow_caching:
                self._write_to_cache(expr, res)
        return res
