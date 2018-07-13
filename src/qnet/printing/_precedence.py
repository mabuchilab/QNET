"""A module providing information about the necessity of parenthesis when
printing"""
__all__ = []
__private__ = [
    'PRECEDENCE_VALUES', 'precedence_ScalarTimesX',
    'precedence_OperatorTimesKet', 'precedence_Bra',
    'precedence_SuperOperatorTimesOperator', 'precedence']

from sympy.printing.precedence import (
    PRECEDENCE,
    PRECEDENCE_VALUES as SYMPY_PRECEDENCE_VALUES,
    PRECEDENCE_FUNCTIONS as SYMPY_PRECEDENCE_FUNCTIONS)

# A dictionary assigning precedence values to certain classes. These values are
# treated like they were inherited, so not every single class has to be named
# here.
PRECEDENCE_VALUES = {
    "QuantumPlus": PRECEDENCE["Add"],
    "QuantumTimes": PRECEDENCE["Mul"],
    "ScalarTimesQuantumExpression": PRECEDENCE["Mul"],
    "QuantumAdjoint": PRECEDENCE["Pow"],
    "QuantumDerivative": PRECEDENCE["Mul"] - 1,
    "OperatorPlus": PRECEDENCE["Add"],
    "OperatorTimes": PRECEDENCE["Mul"],
    "ScalarTimesOperator": PRECEDENCE["Mul"],
    "Commutator": PRECEDENCE["Mul"],
    "SingleOperatorOperation": PRECEDENCE["Atom"],
    "OperatorPlusMinusCC": PRECEDENCE["Add"] - 1,
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
    "ScalarPlus": PRECEDENCE["Add"],
    "ScalarTimes": PRECEDENCE["Mul"],
    "ScalarPower": PRECEDENCE["Pow"],
    "PseudoInverse": PRECEDENCE["Atom"] + 1,
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


def precedence_ScalarValue(expr):
    return precedence(expr.val)


PRECEDENCE_FUNCTIONS = {
    "ScalarTimesOperator": precedence_ScalarTimesX,
    "ScalarTimesKet": precedence_ScalarTimesX,
    "OperatorTimesKet": precedence_OperatorTimesKet,
    "Bra": precedence_Bra,
    "ScalarTimesSuperOperator": precedence_ScalarTimesX,
    "SuperOperatorTimesOperator": precedence_SuperOperatorTimesOperator,
    "ScalarValue": precedence_ScalarValue,
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
