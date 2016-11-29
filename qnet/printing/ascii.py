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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################
from .base import Printer
from ..algebra.singleton import Singleton, singleton_object


@singleton_object
class AsciiPrinter(Printer, metaclass=Singleton):
    """Printer for a string (ASCII) representation. See class:`Printer` for
    details"""
    _registry = {}


def ascii(expr):
    """Return an ascii textual representation of the given object /
    expression"""
    return AsciiPrinter.render(expr)
