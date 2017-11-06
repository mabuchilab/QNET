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
from .precedence import precedence, PRECEDENCE
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
        _parenth_left (str): String to use for a left parenthesis
            (e.g. '\left(' in LaTeX). Used by :meth:`_split_op`
        _parenth_left (str): String to use for a right parenthesis
        _dagger_sym (str): Symbol that indicates the complex conjugate of an
            operator. Used by :meth:`_split_op`
        _tensor_sym (str): Symbol to use for tensor products. Used by
            :meth:`_render_hs_label`.

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

    _parenth_left = '('
    _parenth_right = ')'
    _dagger_sym = 'H'
    _tensor_sym = '*'

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

    def _split_identifier(self, identifier):
        """Split the given identifier at the first underscore into (rendered)
        name and subscript. Both `name` and `subscript` are rendered as
        strings"""
        try:
            name, subscript = identifier.split("_", 1)
        except (TypeError, ValueError):
            name = identifier
            subscript = ''
        return self._str(name), self._str(subscript)

    def _split_op(
            self, identifier, hs_label=None, dagger=False, args=None):
        """Return `name`, total `subscript`, total `superscript` and
        `arguments` str. All of the returned strings are fully rendered.

        Args:
            identifier (str): An (non-rendered/ascii) identifier that may
                include a subscript. The output `name` will be the `identifier`
                without any subscript
            hs_label (str): The rendered label for the Hilbert space of the
                operator, or None. Returned unchanged.
            dagger (bool): Flag to indicate whether the operator is daggered.
                If True, :attr:`dagger_sym` will be included in the
                `superscript` (or  `subscript`, depending on the settings)
            args (list or None): List of arguments (expressions). Each element
                will be rendered with :meth:`doprint`. The total list of args
                will then be joined with commas, enclosed
                with :attr:`_parenth_left` and :attr:`parenth_right`, and
                returnd as the `arguments` string
        """
        name, total_subscript = self._split_identifier(identifier)
        total_superscript = ''
        if (hs_label not in [None, '']):
            if self._settings['show_hilbert_space'] == 'subscript':
                if len(total_subscript) == 0:
                    total_subscript = '(' + hs_label + ')'
                else:
                    total_subscript += ',(' + hs_label + ')'
            else:
                total_superscript += '(' + hs_label + ')'
        if dagger:
            total_superscript += self._dagger_sym
        args_str = ''
        if (args is not None) and (len(args) > 0):
            args_str = (self._parenth_left +
                        ",".join([self.doprint(arg) for arg in args]) +
                        self._parenth_right)
        return name, total_subscript, total_superscript, args_str

    def _render_hs_label(self, hs):
        """Return the label of the given Hilbert space as a string"""
        if isinstance(hs.__class__, Singleton):
            return self._str(hs.label)
        else:
            return self._tensor_sym.join(
                [self._str(ls.label) for ls in hs.local_factors])

    def _render_op(
            self, identifier, hs=None, dagger=False, args=None, superop=False):
        """Render an operator

        Args:
            identifier (str): The identifier (name/symbol) of the operator. May
                include a subscript, denoted by '_'.
            hs (qnet.algebra.hilbert_space_algebra.HilbertSpace): The Hilbert
                space in which the operator is defined
            dagger (bool): Whether the operator should be daggered
            args (list): A list of expressions that will be rendered with
                :meth:`doprint`, joined with commas, enclosed in parenthesis
            superop (bool): Whether the operator is a super-operator
        """
        hs_label = None
        if hs is not None and self._settings['show_hilbert_space']:
            hs_label = self._render_hs_label(hs)
        name, total_subscript, total_superscript, args_str \
            = self._split_op(identifier, hs_label, dagger, args)
        res = name
        if len(total_subscript) > 0:
            res += "_" + total_subscript
        if len(total_superscript) > 0:
            res += "^" + total_superscript
        if len(args_str) > 0:
            res += args_str
        return res

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

    def parenthesize(self, expr, level, strict=False):
        """Render `expr` and wrap the result in parentheses if the precedence
        of `expr` is below the given `level` (or at the given `level` if
        `strict` is True"""
        needs_parenths = (
            (precedence(expr) < level) or
            ((not strict) and precedence(expr) <= level))
        if needs_parenths:
            return (
                self._parenth_left + self.doprint(expr) + self._parenth_right)
        else:
            return self.doprint(expr)

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
        is_cached, res = self._get_from_cache(expr)
        if not is_cached:
            if isinstance(expr, SympyBasic):
                self._sympy_printer._print_level = self._print_level + 1
                res = self._sympy_printer.doprint(expr)
            else:
                # the _print method, inherited from SympyPrinter implements the
                # internal dispatcher for (3-5)
                res = self._str(self._print(expr))
            self._write_to_cache(expr, res)
        return res

    def _print_CircuitSymbol(self, expr):
        return self._str(expr.name)

    def _print_CPermutation(self, expr):
        return r'Perm(%s)' % (
                ", ".join(map(self._str, expr.permutation)))

    def _print_SeriesProduct(self, expr):
        prec = precedence(expr)
        return " << ".join(
            [self.parenthesize(op, prec) for op in expr.operands])

    def _print_Concatenation(self, expr):
        prec = precedence(expr)
        reduced_operands = []  # reduce consecutive identities to a str
        id_count = 0
        for o in expr.operands:
            if self._isinstance(o, 'CIdentity'):
                id_count += 1
            else:
                if id_count > 0:
                    reduced_operands.append(
                        "cid({cdim}".format(cdim=id_count))
                    id_count = 0
                reduced_operands.append(o)
        return " + ".join(
            [self.parenthesize(op, prec) for op in reduced_operands])

    def _print_Feedback(self, expr):
        o, i = expr.out_in_pair
        return '[{operand}]_{{{output}->{input}}}'.format(
            operand=self.doprint(expr.operand), output=o, input=i)

    def _print_SeriesInverse(self, expr):
        return r'[{operand}]^{{-1}}'.format(
            operand=self.doprint(expr.operand))

    def _print_OperatorSymbol(self, expr, adjoint=False):
        return self._render_op(expr.identifier, expr._hs, dagger=adjoint)

    def _print_LocalOperator(self, expr, adjoint=False):
        if adjoint:
            dagger = not expr._dagger
        else:
            dagger = expr._dagger
        return self._render_op(
            expr._identifier, expr._hs, dagger=dagger, args=expr.args)

    def _print_LocalSigma(self, expr, adjoint=False):
        if self._settings['local_sigma_as_ketbra']:
            if adjoint:
                res = "|%s><%s|" % (expr.k, expr.j)
            else:
                res = "|%s><%s|" % (expr.j, expr.k)
            if self._settings['show_hilbert_space']:
                hs_label = self._render_hs_label(expr._hs)
                if self._settings['show_hilbert_space'] == 'subscript':
                    res += '_(%s)' % hs_label
                else:
                    res += '^(%s)' % hs_label
                return res
        else:
            if expr._is_projector:
                identifier = "%s_%s" % (expr._identifier, expr.j)
            else:
                if adjoint:
                    identifier = "%s_%s,%s" % (
                        expr._identifier, expr.k, expr.j)
                else:
                    identifier = "%s_%s,%s" % (
                        expr._identifier, expr.j, expr.k)
            return self._render_op(identifier, expr._hs, dagger=adjoint)

    def _print_IdentityOperator(self, expr):
        return "1"

    def _print_ZeroOperator(self, expr):
        return "0"

    def _print_OperatorPlus(self, expr, adjoint=False):
        prec = precedence(expr)
        l = []
        for term in expr.args:
            t = self.doprint(term)
            if t.startswith('-'):
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < prec:
                l.extend([sign, self._parenth_left + t + self._parenth_right])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_OperatorTimes(self, expr):
        prec = precedence(expr)
        return " * ".join(
            [self.parenthesize(op, prec) for op in expr.operands])

    def _print_ScalarTimesOperator(self, expr, product_sym=" * "):
        prec = PRECEDENCE['Mul']
        coeff, term = expr.coeff, expr.term
        term_str = self.doprint(term)
        if precedence(term) < prec:
            term_str = self._parenth_left + term_str + self._parenth_right

        if coeff == -1:
            if term_str.startswith(self._parenth_left):
                return "- " + term_str
            else:
                return "-" + term_str
        coeff_str = self.doprint(coeff)

        if term_str == '1':
            return coeff_str
        else:
            coeff_str = coeff_str.strip()
            if precedence(coeff) < prec and precedence(-coeff) < prec:
                # the above precedence check catches on only for true sums
                coeff_str = (
                    self._parenth_left + coeff_str + self._parenth_right)
            return coeff_str + product_sym + term_str.strip()

    def _print_Commutator(self, expr):
        return "[" + self.doprint(expr.A) + ", " + self.doprint(expr.B) + "]"

    def _print_OperatorTrace(self, expr):
        s = self._render_hs_label(expr._over_space)
        o = self.doprint(expr.operand)
        return r'tr_({space})[{operand}]'.format(space=s, operand=o)

    def _print_Adjoint(self, expr, adjoint=False):
        o = expr.operand
        if self._isinstance(o, 'LocalOperator'):
            if adjoint:
                dagger = o._dagger
            else:
                dagger = not o._dagger
            return self._render_op(
                o.identifier, hs=o.space, dagger=dagger, args=o.args[1:])
        elif self._isinstance(o, 'OperatorSymbol'):
            return self._render_op(
                o.identifier, hs=o.space, dagger=(not adjoint))
        else:
            if adjoint:
                return self.doprint(o)
            else:
                return (
                    self._parenth_left + self.doprint(o) +
                    self._parenth_right + "^" + self._dagger_sym)

    def _print_OperatorPlusMinusCC(self, expr):
        prec = precedence(expr)
        o = expr.operand
        sign_str = ' + '
        if expr._sign < 0:
            sign_str = ' - '
        return self.parenthesize(o, prec) + sign_str + "c.c."

    def _print_PseudoInverse(self, expr):
        prec = precedence(expr)
        return self.parenthesize(expr.operand, prec) + "^+"

    def _print_NullSpaceProjector(self, expr, adjoint=False):
        null_space_proj_sym = 'P_Ker'
        return self._render_op(
            null_space_proj_sym, hs=None, args=expr.operands, dagger=adjoint)


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
            kwargs = ", ".join(
                        ["%s=%s" % (key, key_sub_render(expr.kwargs[key]))
                            for key in keys])
            if len(args) > 0:
                kwargs = ", " + kwargs
        return head_repr_fmt.format(
            head=expr.__class__.__name__,
            args=", ".join([sub_render(arg) for arg in args]),
            kwargs=kwargs)
    elif isinstance(expr, SympyBasic):
        return sympy_srepr(expr)
    else:
        return repr(expr)
