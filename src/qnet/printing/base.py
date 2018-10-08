"""Provides the base class for Printers"""

from sympy.core.basic import Basic as SympyBasic
from sympy.printing.printer import Printer as SympyPrinter

from ..algebra.core.scalar_algebra import Scalar
from ..utils.indices import StrLabel
from .sympy import SympyStrPrinter
from ._render_head_repr import render_head_repr

__all__ = []
__private__ = ['QnetBasePrinter']


class QnetBasePrinter(SympyPrinter):
    """Base class for all QNET expression printers

    Args:
        cache (dict or None): A dict that maps expressions to strings. It may
            be given during istantiation to use pre-defined strings for
            specific expressions. The cache will be updated as the printer is
            used.
        settings (dict or None): A dict of settings.

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
            expression that is already in `cache`, the result from the
            cache is returned.
        _sympy_printer (sympy.printing.printer.Printer): The printer instance
            that will be used to print any Sympy expression.
        _allow_caching (bool): A flag that may be set to completely disable
            caching
        _print_level (int): The recursion depth of :meth:`doprint`
            (>= 1 inside any of the ``_print*`` methods)

    Raises:
        TypeError: If any key in `settings` is not defined in the
            `_default_settings` of the printer, respectively the
            `sympy_printer_cls`.
    """

    sympy_printer_cls = SympyStrPrinter

    _default_settings = {}
    # note that subclasses should always define their own _default_settings

    _allow_caching = True

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
        if settings is not None:
            for key in settings:
                key_ok = False
                if key in self.sympy_printer_cls._default_settings:
                    key_ok = True
                    sympy_settings[key] = settings[key]
                if key in self._default_settings:
                    key_ok = True
                    qnet_settings[key] = settings[key]
                if not key_ok:
                    raise TypeError(
                        "%s is not a valid setting for either %s or %s" % (
                            key, self.__class__.__name__,
                            self.sympy_printer_cls.__name__))
        self._sympy_printer = self.sympy_printer_cls(settings=sympy_settings)
        super().__init__(settings=qnet_settings)

    def _render_str(self, string):
        if isinstance(string, StrLabel):
            string = string._render(string.expr)
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
        else:  # numeric type
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
            res = self._print(expr, *args, **kwargs)
        return res

    def doprint(self, expr, *args, **kwargs):
        """Returns printer's representation for expr (as a string)

        The representation is obtained by the following methods:

        1. from the :attr:`cache`
        2. If `expr` is a Sympy object, delegate to the
           :meth:`~sympy.printing.printer.Printer.doprint` method of
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
            if isinstance(expr, Scalar._val_types):
                res = self._print_SCALAR_TYPES(expr, *args, **kwargs)
            elif isinstance(expr, str):
                return self._render_str(expr)
            else:
                # the _print method, inherited from SympyPrinter implements the
                # internal dispatcher for (3-5)
                res = self._str(self._print(expr, *args, **kwargs))
            if allow_caching:
                self._write_to_cache(expr, res)
        return res
