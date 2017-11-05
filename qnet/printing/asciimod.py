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

    def _print_CircuitSymbol(self, expr):
        return self._str(expr.name)

    def _print_CPermutation(self, expr):
        return r'Perm(%s)' % (
                ", ".join(map(self._str, self.permutation)))

    def _print_SeriesProduct(self, expr):
        raise NotImplementedError  # TODO

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
                    self._parenth_left + self.doprin(o) + self._parenth_right +
                    "^" + self._dagger_sym)


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
