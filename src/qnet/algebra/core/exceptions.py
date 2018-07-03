"""Exceptions and Errors raised by QNET"""


__all__ = [
    'AlgebraException', 'AlgebraError', 'InfiniteSumError', 'CannotSimplify',
    'CannotConvertToSLH', 'CannotVisualize', 'WrongCDimError',
    'IncompatibleBlockStructures', 'CannotEliminateAutomatically',
    'BasisNotSetError', 'UnequalSpaces', 'OverlappingSpaces',
    'SpaceTooLargeError', 'CannotSymbolicallyDiagonalize',
    'BadLiouvillianError', 'NonSquareMatrix', 'NoConjugateMatrix']


class AlgebraException(Exception):
    """Base class for all algebraic exceptions"""
    pass


class AlgebraError(AlgebraException):
    """Base class for all algebraic errors"""
    pass


class InfiniteSumError(AlgebraError):
    """Raised when expanding a sum into an infinite number of terms"""
    pass


class CannotSimplify(AlgebraException):
    """Raised when a rule cannot further simplify an expression"""
    pass


class CannotConvertToSLH(AlgebraException):
    """Raised when a circuit algebra object cannot be converted to SLH"""


class CannotVisualize(AlgebraException):
    """Raised when a circuit cannot be visually represented."""


class WrongCDimError(AlgebraError):
    """Raised for mismatched channel number in circuit series"""


class IncompatibleBlockStructures(AlgebraError):
    """Raised for invalid block-decomposition

    This is raised when a circuit decomposition into a block-structure is
    requested that is icompatible with the actual block structure of the
    circuit expression."""


class CannotEliminateAutomatically(AlgebraError):
    """Raised when attempted automatic adiabatic elimination fails."""


class BasisNotSetError(AlgebraError):
    """Raised if the basis or a Hilbert space dimension is unavailable"""


class UnequalSpaces(AlgebraError):
    """Raised when objects fail to be in the same Hilbert space.

    This happens for example when trying to add two states from different
    Hilbert spaces."""


class OverlappingSpaces(AlgebraError):
    """Raised when objects fail to be in separate Hilbert spaces."""


class SpaceTooLargeError(AlgebraError):
    """Raised when objects fail to be have overlapping Hilbert spaces."""


class CannotSymbolicallyDiagonalize(AlgebraException):
    """Matrix cannot be diagonalized analytically.

    Signals that a fallback to numerical diagonalization is required.
    """


class BadLiouvillianError(AlgebraError):
    """Raised when a Liouvillian is not of standard Lindblad form."""
    pass


class NonSquareMatrix(AlgebraError):
    """Raised when a :class:`.Matrix` fails to be square"""


class NoConjugateMatrix(AlgebraError):
    """Raised when entries of :class:`.Matrix` have no defined conjugate"""
