"""Exceptions and Errors raised by QNET"""


__all__ = [
    'AlgebraException', 'AlgebraError', 'InfiniteSumError', 'CannotSimplify',
    'WrongSignatureError', 'CannotConvertToSLH', 'CannotConvertToABCD',
    'CannotVisualize', 'WrongCDimError', 'IncompatibleBlockStructures',
    'CannotEliminateAutomatically', 'BasisNotSetError', 'UnequalSpaces',
    'OverlappingSpaces', 'SpaceTooLargeError', 'CannotSymbolicallyDiagonalize',
    'BadLiouvillianError', 'NonSquareMatrix']


class AlgebraException(Exception):
    """Base class for all errors concerning the mathematical definitions and
    rules of an algebra."""
    pass


class AlgebraError(AlgebraException):
    """Base class for all errors concerning the mathematical definitions and
    rules of an algebra."""
    pass


class InfiniteSumError(AlgebraError):
    """Raised when expanding a sum into an infinite number of terms"""
    pass


class CannotSimplify(AlgebraException):
    """Raised when an expression cannot be further simplified"""
    pass


class WrongSignatureError(AlgebraError):
    """Raised when an operation is instantiated with operands of the wrong
    signature."""
    pass


class CannotConvertToSLH(AlgebraException):
    """Is raised when a circuit algebra object cannot be converted to a
    concrete SLH object."""


class CannotConvertToABCD(AlgebraException):
    """Is raised when a circuit algebra object cannot be converted to a
    concrete ABCD object."""


class CannotVisualize(AlgebraException):
    """Is raised when a circuit algebra object cannot be visually
    represented."""


class WrongCDimError(AlgebraError):
    """Is raised when two object are tried to joined together in series but
    have different channel dimensions."""


class IncompatibleBlockStructures(AlgebraError):
    """Is raised when a circuit decomposition into a block-structure is
    requested that is icompatible with the actual block structure of the
    circuit expression."""


class CannotEliminateAutomatically(AlgebraError):
    """Raised when attempted automatic adiabatic elimination fails."""


class BasisNotSetError(AlgebraError):
    """Raised if the basis or a Hilbert space dimension is requested but is not
    available"""


class UnequalSpaces(AlgebraError):
    pass


class OverlappingSpaces(AlgebraError):
    pass


class SpaceTooLargeError(AlgebraError):
    pass


class CannotSymbolicallyDiagonalize(AlgebraException):
    pass


class BadLiouvillianError(AlgebraError):
    """Raise when a Liouvillian is not of standard Lindblad form."""
    pass


class NonSquareMatrix(Exception):
    pass
