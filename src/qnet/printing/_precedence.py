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
"""A module providing information about the necessity of parenthesis when
printing"""

from sympy.printing.precedence import (
    PRECEDENCE,
    PRECEDENCE_VALUES as SYMPY_PRECEDENCE_VALUES,
    PRECEDENCE_FUNCTIONS as SYMPY_PRECEDENCE_FUNCTIONS)

# A dictionary assigning precedence values to certain classes. These values are
# treated like they were inherited, so not every single class has to be named
# here.
PRECEDENCE_VALUES = {
    "OperatorPlus": PRECEDENCE["Add"],
    "OperatorTimes": PRECEDENCE["Mul"],
    "ScalarTimesOperator": PRECEDENCE["Mul"],
    "Commutator": PRECEDENCE["Add"],
    "SingleOperatorOperation": PRECEDENCE["Atom"],
    "OperatorPlusMinusCC": PRECEDENCE["Add"] - 1,
    "OperatorCommutator": PRECEDENCE["Atom"],
    "SeriesProduct": PRECEDENCE["Mul"],
    "Concatenation": PRECEDENCE["Add"],
    "SeriesInverse": PRECEDENCE["Atom"],
    "KetPlus": PRECEDENCE["Add"],
    "IndexedSum": PRECEDENCE["Add"],
    "TensorKet": PRECEDENCE["Mul"],
    "BraKet": PRECEDENCE["Mul"],
    "KetBra": PRECEDENCE["Mul"],
    "Adjoint": PRECEDENCE["Pow"],
    "SuperOperatorPlus": PRECEDENCE["Add"],
    "SuperOperatorTimes": PRECEDENCE["Mul"],
    "SuperAdjoint": PRECEDENCE["Pow"],
}
PRECEDENCE_VALUES.update(SYMPY_PRECEDENCE_VALUES)

# Sometimes it's not enough to assign a fixed precedence value to a
# class. Then a function can be inserted in this dictionary that takes
# an instance of this class as argument and returns the appropriate
# precedence value.

# Precedence functions


def precedence_ScalarTimesX(expr):
    # TODO: can we avoid rendering expr.coeff, cf.
    # from sympy.core.function import _coeff_isneg
    if str(expr.coeff).startswith('-'):
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_OperatorTimesKet(expr):
    if str(expr.operator).startswith('-'):
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Bra(expr):
    return precedence(expr.ket)


def precedence_SuperOperatorTimesOperator(expr):
    if str(expr.sop).startswith('-'):
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


PRECEDENCE_FUNCTIONS = {
    "ScalarTimesOperator": precedence_ScalarTimesX,
    "ScalarTimesKet": precedence_ScalarTimesX,
    "OperatorTimesKet": precedence_OperatorTimesKet,
    "Bra": precedence_Bra,
    "ScalarTimesSuperOperator": precedence_ScalarTimesX,
    "SuperOperatorTimesOperator": precedence_SuperOperatorTimesOperator,
}
PRECEDENCE_FUNCTIONS.update(SYMPY_PRECEDENCE_FUNCTIONS)


def precedence(item):
    """Returns the precedence of a given object."""
    try:
        mro = item.__class__.__mro__
    except AttributeError:
        return PRECEDENCE["Atom"]
    for i in mro:
        n = i.__name__
        if n in PRECEDENCE_FUNCTIONS:
            return PRECEDENCE_FUNCTIONS[n](item)
        elif n in PRECEDENCE_VALUES:
            return PRECEDENCE_VALUES[n]
    return PRECEDENCE["Atom"]
