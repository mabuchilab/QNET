# This file is part of QNET.
#
# QNET is free software: you can redistribute it and/or modify
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
"""Matrices of Operators"""

from numpy import (
        array as np_array, ndarray, conjugate as np_conjugate,
        hstack as np_hstack, vstack as np_vstack, zeros as np_zeros, ones as
        np_ones, diag as np_diag)
from sympy import Basic as SympyBasic, I, sympify

from .abstract_algebra import (
        Expression, Operation, SCALAR_TYPES, cache_attr, substitute)
from .operator_algebra import Operator, scalar_free_symbols, simplify_scalar
from .hilbert_space_algebra import TrivialSpace, ProductSpace
from .permutations import check_permutation

__all__ = [
    'NonSquareMatrix', 'Matrix', 'block_matrix', 'diagm', 'hstackm',
    'identity_matrix', 'permutation_matrix', 'vstackm', 'zerosm',
    'ImMatrix', 'ReMatrix', 'ImAdjoint', 'ReAdjoint']

__private__ = [  # anything not in __all__ must be in __private__
    'sympyOne', 'Re', 'Im']

sympyOne = sympify(1)


class NonSquareMatrix(Exception):
    pass


class Matrix(Expression):
    """Matrix with Operator (or scalar-) valued elements."""
    matrix = None
    _hash = None

    def __init__(self, m):
        if isinstance(m, ndarray):
            self.matrix = m
        elif isinstance(m, Matrix):
            self.matrix = np_array(m.matrix)
        else:
            self.matrix = np_array(m)
        if len(self.matrix.shape) < 2:
            self.matrix = self.matrix.reshape((self.matrix.shape[0], 1))
        if len(self.matrix.shape) > 2:
            raise ValueError()
        super().__init__(self.matrix)

    @property
    def shape(self):
        """The shape of the matrix ``(nrows, ncols)``"""
        return self.matrix.shape

    @property
    def block_structure(self):
        """For square matrices this gives the block (-diagonal) structure of
        the matrix as a tuple of integers that sum up to the full dimension.

        :type: tuple
        """
        n, m = self.shape
        if n != m:
            raise AttributeError("block_structure only defined for square "
                                 "matrices")
        for k in range(1, n):
            if ((self.matrix[:k, k:] == 0).all() and
                    (self.matrix[k:, :k] == 0).all()):
                return (k,) + self[k:, k:].block_structure
        return n,

    def _get_blocks(self, block_structure):
        n, m = self.shape
        if n == m:
            if not sum(block_structure) == n:
                raise ValueError()
            if not len(block_structure):
                return ()
            j = block_structure[0]

            if ((self.matrix[:j, j:] == 0).all() and
                    (self.matrix[j:, :j] == 0).all()):
                return ((self[:j, :j],) +
                        self[j:, j:]._get_blocks(block_structure[1:]))
            else:
                raise ValueError()
        elif m == 1:
            if not len(block_structure):
                return ()
            else:
                return ((self[:block_structure[0], :],) +
                        self[:block_structure[0], :]
                        ._get_blocks(block_structure[1:]))
        else:
            raise ValueError()

    @property
    def args(self):
        return (self.matrix, )

    @property
    def is_zero(self):
        """Are all elements of the matrix zero?"""
        return (self.matrix == 0).all()

    @classmethod
    def _instance_key(cls, args, kwargs):
        matrix = args[0]
        return (cls, tuple(matrix.flatten()), tuple(matrix.shape))

    def __hash__(self):
        if not self._hash:
            self._hash = hash((tuple(self.matrix.flatten()),
                               self.matrix.shape, Matrix))
        return self._hash

    def __eq__(self, other):
        return (isinstance(other, Matrix) and
                self.shape == other.shape and
                (self.matrix == other.matrix).all())

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.matrix + other.matrix)
        else:
            return Matrix(self.matrix + other)

    def __radd__(self, other):
        return Matrix(other + self.matrix)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.matrix.dot(other.matrix))
        else:
            return Matrix(self.matrix * other)

    def __rmul__(self, other):
        return Matrix(other * self.matrix)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self * (sympyOne / other)
        raise NotImplementedError("Can't divide matrix %s by %s"
                                  % (self, other))

    __truediv__ = __div__

    #    def __pow__(self, power):
    #        return OperatorMatrix(self.matrix.__pow__(power))

    def transpose(self):
        """The transpose matrix"""
        return Matrix(self.matrix.T)

    def conjugate(self):
        """The element-wise conjugate matrix, i.e., if an element is an
        operator this means the adjoint operator, but no transposition of
        matrix elements takes place.
        """
        return Matrix(np_conjugate(self.matrix))

    @property
    def T(self):
        """Transpose matrix"""
        return self.transpose()

    def adjoint(self):
        """Return the adjoint operator matrix, i.e. transpose and the Hermitian
        adjoint operators of all elements.
        """
        return self.T.conjugate()

    dag = adjoint

    def _render(self, fmt, adjoint=False):
        assert not adjoint, "adjoint not defined"
        printer = getattr(self, "_"+fmt+"_printer")
        row_strs = []
        if len(self.matrix) == 0:
            row_strs.append(
                    printer.matrix_row_left_sym +
                    printer.matrix_row_right_sym)
            row_strs.append(
                    printer.matrix_row_left_sym +
                    printer.matrix_row_right_sym)
        else:
            for row in self.matrix:
                row_strs.append(
                        printer.matrix_row_left_sym +
                        printer.matrix_col_sep_sym.join(
                            [printer.render(entry) for entry in row]) +
                        printer.matrix_row_right_sym)
        return (printer.matrix_left_sym +
                printer.matrix_row_sep_sym.join(row_strs) +
                printer.matrix_right_sym)

    def trace(self):
        if self.shape[0] == self.shape[1]:
            return sum(self.matrix[k, k] for k in range(self.shape[0]))
        raise NonSquareMatrix(repr(self))

    @property
    def H(self):
        """Return the adjoint operator matrix, i.e. transpose and the Hermitian
        adjoint operators of all elements."""
        return self.adjoint()

    def __getitem__(self, item_id):
        item = self.matrix.__getitem__(item_id)
        if isinstance(item, ndarray):
            return Matrix(item)
        return item

    def element_wise(self, method):
        """Apply a method to each matrix element and return the result in a new
        operator matrix of the same shape.  :param method: A method taking a
        single argument.
        :type method: FunctionType
        :return: Operator matrix with results of method applied element-wise.
        :rtype: Matrix
        """
        s = self.shape
        emat = [method(o) for o in self.matrix.flatten()]
        return Matrix(np_array(emat).reshape(s))

    def series_expand(self, param, about, order):
        """Expand the matrix expression as a truncated power series in a scalar
        parameter.

        :param param: Expansion parameter.
        :type param: sympy.core.symbol.Symbol
        :param about: Point about which to expand.
        :type about:  Any one of Operator.scalar_types
        :param order: Maximum order of expansion.
        :type order: int >= 0
        :return: tuple of length (order+1), where the entries are the expansion coefficients.
        :rtype: tuple of Operator
        """
        s = self.shape
        emats = zip(*[o.series_expand(param, about, order)
                      for o in self.matrix.flatten()])
        return tuple((Matrix(np_array(em).reshape(s)) for em in emats))

    def expand(self):
        """Expand each matrix element distributively.
        :return: Expanded matrix.
        :rtype: Matrix
        """
        m = lambda o: o.expand() if isinstance(o, Operator) else o
        return self.element_wise(m)

    def _substitute(self, var_map):

        def _substitute(o):
            sympy_var_map = {k: v for (k, v) in var_map.items()
                             if isinstance(k, SympyBasic)}
            if isinstance(o, Operation):
                return substitute(o, var_map)
            elif isinstance(o, SympyBasic):
                return o.subs(sympy_var_map)
            else:
                return o

        return self.element_wise(_substitute)

    def all_symbols(self):
        ret = set()
        for o in self.matrix.flatten():
            if isinstance(o, Operator):
                ret = ret | o.all_symbols()
            else:
                ret = ret | scalar_free_symbols(o)
        return ret


    @property
    def space(self):
        """Combined Hilbert space of all matrix elements."""
        arg_spaces = [o.space for o in self.matrix.flatten()
                      if hasattr(o, 'space')]
        if len(arg_spaces) == 0:
            return TrivialSpace
        else:
            return ProductSpace.create(*arg_spaces)

    def simplify_scalar(self):
        """
        Simplify all scalar expressions appearing in the Matrix.
        """
        return self.element_wise(simplify_scalar)


def hstackm(matrices):
    """Generalizes `numpy.hstack` to OperatorMatrix objects."""
    return Matrix(np_hstack(tuple(m.matrix for m in matrices)))


def vstackm(matrices):
    """Generalizes `numpy.vstack` to OperatorMatrix objects."""
    arr = np_vstack(tuple(m.matrix for m in matrices))
    #    print(tuple(m.matrix.dtype for m in matrices))
    #    print(arr.dtype)
    return Matrix(arr)


def diagm(v, k=0):
    """Generalizes the diagonal matrix creation capabilities of `numpy.diag` to
    OperatorMatrix objects."""
    return Matrix(np_diag(v, k))


def block_matrix(A, B, C, D):
    r"""Generate the operator matrix with quadrants

    .. math::

       \begin{pmatrix} A B \\ C D \end{pmatrix}

    :param A: Matrix of shape ``(n, m)``
    :type A: Matrix
    :param B: Matrix of shape ``(n, k)``
    :type B: Matrix
    :param C: Matrix of shape ``(l, m)``
    :type C: Matrix
    :param D: Matrix of shape ``(l, k)``
    :type D: Matrix

    :return: The combined block matrix [[A, B], [C, D]].
    :type: OperatorMatrix
    """
    return vstackm((hstackm((A, B)), hstackm((C, D))))


def identity_matrix(N):
    """Generate the N-dimensional identity matrix.

    :param N: Dimension
    :type N: int
    :return: Identity matrix in N dimensions
    :rtype: Matrix
    """
    return diagm(np_ones(N, dtype=int))


def zerosm(shape, *args, **kwargs):
    """Generalizes ``numpy.zeros`` to :py:class:`Matrix` objects."""
    return Matrix(np_zeros(shape, *args, **kwargs))


def permutation_matrix(permutation):
    r"""Return an orthogonal permutation matrix
    :math:`M_\sigma`
    for a permutation :math:`\sigma` defined by the image tuple
    :math:`(\sigma(1), \sigma(2),\dots \sigma(n))`,
    such that

    .. math::

        M_\sigma \vec{e}_i = \vec{e}_{\sigma(i)}

    where :math:`\vec{e}_k` is the k-th standard basis vector.
    This definition ensures a composition law:

    .. math::

        M_{\sigma \cdot \tau} = M_\sigma M_\tau.

    The column form of :math:`M_\sigma` is thus given by

    .. math::

        M = (\vec{e}_{\sigma(1)}, \vec{e}_{\sigma(2)}, \dots \vec{e}_{\sigma(n)}).

    :param permutation: A permutation image tuple (zero-based indices!)
    :type permutation: tuple
    """
    assert check_permutation(permutation)
    n = len(permutation)
    op_matrix = np_zeros((n, n), dtype=int)
    for i, j in enumerate(permutation):
        op_matrix[j, i] = 1
    return Matrix(op_matrix)


def Im(op):
    """The imaginary part of a number or operator. Acting on OperatorMatrices,
    it produces the element-wise imaginary parts.

    :param op: Anything that has a conjugate method.
    :type op: Operator or Matrix or any of Operator.scalar_types
    :return: The imaginary part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() - op) * I / 2


ImMatrix = Im  # for flat API


def Re(op):
    """The real part of a number or operator. Acting on OperatorMatrices, it
    produces the element-wise real parts.

    :param op: Anything that has a conjugate method.
    :type op: Operator or Matrix or any of Operator.scalar_types
    :return: The real part of the operand.
    :rtype: Same as type of `op`.
    """
    return (op.conjugate() + op) / 2


ReMatrix = Re  # for flat API


def ImAdjoint(opmatrix):
    """
    The imaginary part of an OperatorMatrix, i.e. a Hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: Matrix
    :return: The matrix imaginary part of the operand.
    :rtype: Matrix
    """
    return (opmatrix.H - opmatrix) * I / 2


def ReAdjoint(opmatrix):
    """The real part of an OperatorMatrix, i.e. a Hermitian OperatorMatrix
    :param opmatrix: The operand.
    :type opmatrix: Matrix
    :return: The matrix real part of the operand.
    :rtype: Matrix
    """
    return (opmatrix.H + opmatrix) / 2
