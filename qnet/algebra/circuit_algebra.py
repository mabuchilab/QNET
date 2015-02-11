# coding=utf-8
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
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

"""
Circuit Algebra
===============

This module defines the circuit algebra for quantum optical feedback and feedforward circuits in the zero-internal time-delay limit.
For more details see :ref:`circuit_algebra`.

References:

.. [1] Gough, James & Nurdin (2010). Squeezing components in linear quantum feedback networks. Physical Review A, 81(2). doi:10.1103/PhysRevA.81.023804
.. [2] Gough & James (2008). Quantum Feedback Networks: Hamiltonian Formulation. Communications in Mathematical Physics, 287(3), 1109-1132. doi:10.1007/s00220-008-0698-8
.. [3] Gough & James (2009). The Series Product and Its Application to Quantum Feedforward and Feedback Networks. IEEE Transactions on Automatic Control, 54(11), 2530-2544. doi:10.1109/TAC.2009.2031205


"""

from __future__ import division

import os

from qnet.algebra.operator_algebra import *
from qnet.algebra.permutations import *

class CannotConvertToSLH(AlgebraException):
    """
    Is raised when a circuit algebra object cannot be converted to a concrete SLH object.
    """


class CannotConvertToABCD(AlgebraException):
    """
    Is raised when a circuit algebra object cannot be converted to a concrete ABCD object.
    """


class CannotVisualize(AlgebraException):
    """
    Is raised when a circuit algebra object cannot be visually represented.
    """


class WrongCDimError(AlgebraError):
    """
    Is raised when two object are tried to joined together in series but have different channel dimensions.
    """


class IncompatibleBlockStructures(AlgebraError):
    """
    Is raised when a circuit decomposition into a block-structure is requested
    that is icompatible with the actual block structure of the circuit expression.
    """

@six.add_metaclass(ABCMeta)
class Circuit(object):
    """
    Abstract base class for the circuit algebra elements.
    """

    @property
    def cdim(self):
        """
        The channel dimension of the circuit expression,
        i.e. the number of external bosonic noises/inputs that the circuit couples to.
        """
        return self._cdim

    @abstractproperty
    def _cdim(self):
        raise NotImplementedError(self.__class__.__name__)

    @property
    def block_structure(self):
        """
        If the circuit is *reducible* (i.e., it can be represented as a :py:class:Concatenation: of individual circuit expressions),
        this gives a tuple of cdim values of the subblocks.
        E.g. if A and B are irreducible and have ``A.cdim = 2``, ``B.cdim = 3``

            >>> A = CircuitSymbol('A', 2)
            >>> B = CircuitSymbol('B', 3)

        Then the block structure of their Concatenation is:

            >>> (A + B).block_structure
                (2,3),

        while

            >>> A.block_structure
                (2,)
            >>> B.block_structure
                (3,)

        """
        return self._block_structure

    @property
    def _block_structure(self):
        return self.cdim,

    def index_in_block(self, channel_index):
        """
        Yield the index a channel has within the subblock it belongs to.
        I.e., only for reducible circuits, this gives a result different from the argument itself.

        :param channel_index: The index of the external channel
        :type channel_index: int
        :return: The index of the external channel within the subblock it belongs to.
        :rtype: int
        :raise: ValueError

        """
        if channel_index < 0 or channel_index >= self.cdim:
            raise ValueError()

        struct = self.block_structure

        if len(struct) == 1:
            return channel_index, 0
        i = 1
        while sum(struct[:i]) <= channel_index and i < self.cdim:
            i += 1
        block_index = i - 1
        index_in_block = channel_index - sum(struct[:block_index])

        return index_in_block, block_index


    def get_blocks(self, block_structure=None):
        """
        For a reducible circuit, get a sequence of subblocks that when concatenated again yield the original circuit.
        The block structure given has to be compatible with the circuits actual block structure,
        i.e. it can only be more coarse-grained.

        :param block_structure: The block structure according to which the subblocks are generated (default = ``None``, corresponds to the circuit's own block structure)
        :type block_structure: tuple
        :return: A tuple of subblocks that the circuit consists of.
        :rtype: tuple
        :raises: IncompatibleBlockStructures
        """
        if block_structure is None:
            #noinspection PyRedeclaration
            block_structure = self.block_structure
        try:
            return self._get_blocks(block_structure)
        except IncompatibleBlockStructures as e:
            raise e

    def _get_blocks(self, block_structure):
        if block_structure == self.block_structure:
            return (self, )
        raise IncompatibleBlockStructures("Requested incompatible block structure %s" % (block_structure,))

    def series_inverse(self):
        """
        Return the inverse object (under the series product) for a circuit.
        In general for any X

            >>> X << X.series_inverse() == X.series_inverse() << X == cid(X.cdim)
                True
        """
        return self._series_inverse()

    def _series_inverse(self):
        return SeriesInverse.create(self)

    def feedback(self, out_index=None, in_index=None):
        """
        Return a circuit with self-feedback from the output port (zero-based) ``out_index`` to the input port ``in_index``.

        :param out_index: The output port from which the feedback connection leaves (zero-based, default = ``None`` corresponds to the *last* port).
        :type out_index: int or NoneType
        :param in_index: The input port into which the feedback connection goes (zero-based, default = ``None`` corresponds to the *last* port).
        :type in_index: int or NoneType
        """
        if out_index is None:
            #noinspection PyRedeclaration
            out_index = self.cdim - 1
        if in_index is None:
            #noinspection PyRedeclaration
            in_index = self.cdim - 1

        return self._feedback(out_index, in_index)

    def _feedback(self, out_index, in_index):
        return Feedback.create(self, out_index, in_index)

    def show(self):
        """
        Show the circuit expression in an IPython notebook.
        """
        from IPython.display import Image, display

        fname = self.render()
        display(Image(filename=fname))


    def render(self, fname=''):
        """
        Render the circuit expression and store the result in a file

        :param fname: Path to an image file to store the result in.
        :type fname: basestring

        :return (str): The path to the image file
        """
        import qnet.misc.circuit_visualization as circuit_visualization

        if not fname:
            from tempfile import gettempdir
            from time import time

            tmp_dir = gettempdir()
            fname = tmp_dir + "/tmp_{}.png".format(hash(time))

        if circuit_visualization.draw_circuit(self, fname):
            done = False
            for k in range(20):
                if os.path.exists(fname):

                    done = True
                    break
                else:
                    time.sleep(.5)
            if done:
                return fname

        raise CannotVisualize()





        #return CircuitVisualizer(self)

    def creduce(self):
        """
        If the circuit is reducible, try to reduce each subcomponent once.
        Depending on whether the components at the next hierarchy-level are themselves reducible,
        successive ``circuit.creduce()`` operations yields an increasingly fine-grained decomposition of a circuit into its most primitive elements.
        """
        return self._creduce()

    @abstractmethod
    def _creduce(self):
        return self

    def toSLH(self):
        """
        Return the SLH representation of a circuit. This can fail if there are un-substituted pure circuit all_symbols (:py:class:`CircuitSymbol`) left
        in the expression or if the circuit includes *non-passive* ABCD models (cf. [1]_)


        """
        return self._toSLH()

    @abstractmethod
    def _toSLH(self):
        raise NotImplementedError(self.__class__.__name__)

    def toABCD(self, linearize=False):
        """
        Return the ABCD representation of a circuit expression. If `linearize=True` all operator expressions giving rise to non-linear equations of motion are dropped.
        This can fail if there are un-substituted pure circuit all_symbols (:py:class:`CircuitSymbol`) left in the expression or if `linearize = False` and the circuit includes non-linear SLH models.
        (cf. [1]_)


        :param linearize: Whether or not to explicitly neglect non-linear contributions (default = False)
        :type linearize: bool
        :return: ABCD model for the circuit
        :rtype: ABCD
        """
        return self._toABCD(linearize)

    @abstractmethod
    def _toABCD(self, linearize):
        raise NotImplementedError(self.__class__.__name__)

    def coherent_input(self, *input_amps):
        """
        Feed coherent input amplitudes into the circuit.
        E.g. For a circuit with channel dimension of two,
        `C.coherent_input(0,1)` leads to an input amplitude of zero into the first and one into the second port.

        :param input_amps: The coherent input amplitude for each port
        :type input_amps: any of :py:attr:`qnet.algebra.operator_algebra.Operator.scalar_types`
        :return: The circuit including the coherent inputs.
        :rtype: Circuit
        :raise: WrongCDimError
        """
        return self._coherent_input(*input_amps)

    def _coherent_input(self, *input_amps):
        if len(input_amps) != self.cdim:
            raise WrongCDimError()
        from qnet.circuit_components.displace_cc import Displace as Displace_cc

        return self << sum([Displace_cc('W', alpha=amp) if amp != 0 else cid(1) for amp in input_amps], cid(0))

    @property
    def space(self):
        """
        All Hilbert space degree of freedoms associated with a given circuit component.
        """
        return self._space

    @abstractproperty
    def _space(self):
        raise NotImplementedError(self.__class__)


    def __lshift__(self, other):
        if isinstance(other, Circuit):
            return SeriesProduct.create(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Circuit):
            return Concatenation.create(self, other)
        return NotImplemented


class SLH(Circuit, Operation):
    """
    SLH class to encapsulate an open system model that is parametrized as described in [2]_ , [3]_ ::

        SLH(S, L, H)

    :param S: The scattering matrix (with in general Operator-valued elements)
    :type S: Matrix
    :param L: The coupling vector (with in general Operator-valued elements)
    :type L: Matrix
    :param H: The internal Hamilton operator
    :type H: Operator
    :raise: ValueError
    """

    #noinspection PyRedeclaration,PyUnresolvedReferences
    def __init__(self, S, L, H):
        if not isinstance(S, Matrix):
            S = Matrix(S)
        if not isinstance(L, Matrix):
            L = Matrix(L)
        if S.shape[0] != L.shape[0]:
            raise ValueError('S and L misaligned: S = {!r}, L = {!r}'.format(S, L))

        #noinspection PyArgumentList
        super(SLH, self).__init__(S, L, H)

    @property
    def _cdim(self):
        return self.S.shape[0]

    def _creduce(self):
        return self

    @property
    def S(self):
        """
        The scattering matrix (with in general Operator-valued elements) of shape ``(cdim, cdim)``

        :rtype: Matrix
        """
        return self.operands[0]

    @property
    def L(self):
        """
        The coupling vector (with in general Operator-valued elements) of shape ``(cdim, 1)``

        :rtype: Matrix
        """
        return self.operands[1]

    #noinspection PyRedeclaration
    @property
    def H(self):
        """
        The internal Hamilton operator.

        :rtype: Operator
        """
        return self.operands[2]


    @property
    def _space(self):
        return self.S.space * self.L.space * space(self.H)


    def series_with_slh(self, other):
        """
        Evaluate the series product with another :py:class:``SLH`` object.

        :param other: An upstream SLH circuit.
        :type other: SLH
        :return: The combines system.
        :rtype: SLH
        """
        new_S = self.S * other.S
        new_L = self.S * other.L + self.L

        delta = ImAdjoint(self.L.adjoint() * self.S * other.L)

        if isinstance(delta, Matrix):
            new_H = self.H + other.H + delta[0, 0]
        else:
            assert delta == 0
            new_H = self.H + other.H

        return SLH(new_S, new_L, new_H)

    def concatenate_slh(self, other):
        """
        Evaluate the concatenation product with another SLH object.

        :param other: Another SLH object
        :type other: SLH
        :return: The combined system
        :rtype: SLH
        """
        selfS = self.S
        otherS = other.S
        new_S = block_matrix(selfS, zerosm((selfS.shape[0], otherS.shape[1]), dtype=int),
                             zerosm((otherS.shape[0], selfS.shape[1]), dtype=int), otherS)
        new_L = vstackm((self.L, other.L))
        new_H = self.H + other.H

        return SLH(new_S, new_L, new_H)

    def __str__(self):
        return "({!s}, {!s}, {!s})".format(self.S, self.L, self.H)

    def _tex(self):
        return r"\left( {}, {}, {} \right)".format(tex(self.S), tex(self.L), tex(self.H))

    def _toSLH(self):
        return self

    def expand(self):
        """
        Expand out all operator expressions within S, L and H and return a new SLH object with these expanded expressions.

        :rtype: SLH
        """
        return SLH(self.S.expand(), self.L.expand(), self.H.expand())

    def simplify_scalar(self):
        """
        Simplify all scalar expressions within S, L and H and return a new SLH object with the simplified expressions.

        :rtype: SLH
        """
        return SLH(self.S.simplify_scalar(), self.L.simplify_scalar(), self.H.simplify_scalar())


    def HL_to_qutip(self, full_space=None):
        """
        Generate and return QuTiP representation matrices for the Hamiltonian and the collapse operators.

        :param full_space: The Hilbert space in which to represent the operators.
        :type full_space: HilbertSpace or None
        :return tuple: (H, [L1, L2, ...]) as numerical qutip.Qobj representations.
        """
        if full_space:
            if not full_space >= self.space:
                raise AlgebraError("full_space = {} needs to at least include self.space = {}".format(str(full_space),
                                                                                                      str(self.space)))
        else:
            full_space = self.space
        H = self.H.to_qutip(full_space)
        Ls = [L.to_qutip(full_space) for L in self.L.matrix.flatten() if isinstance(L, Operator)]

        return H, Ls


    #    _block_structure_ = None
    #
    #    @property
    #    def _block_structure(self):
    #        if self.cdim and not self._block_structure_:
    #            self._block_structure_ = self.S.block_structure
    #        return self._block_structure_
    #
    #    def _get_blocks(self, block_structure):
    #        Sblocks = self.S._get_blocks(block_structure)
    #        Lblocks = self.L._get_blocks(block_structure)
    #        Hblocks = (self.H,) + ((0,)*(len(block_structure)-1))
    #        return tuple(SLH(S,L,H) for (S,L,H) in zip(Sblocks, Lblocks, Hblocks))


    def _series_inverse(self):
        return SLH(self.S.adjoint(), - self.S.adjoint() * self.L, -self.H)

    def _feedback(self, out_index, in_index):
        if not isinstance(self.S, Matrix) or not isinstance(self.L, Matrix):
            return Feedback(self, out_index, in_index)

        n = self.cdim - 1

        if out_index != n:
            return (map_signals_circuit({out_index: n}, self.cdim).toSLH() 
                << self).feedback(in_index=in_index)
        elif in_index != n:
            return (self << map_signals_circuit({n: in_index}, self.cdim).toSLH()).feedback()

        S, L, H = self.operands

        one_minus_Snn = sympyOne - S[n, n]

        if isinstance(one_minus_Snn, Operator):
            if isinstance(one_minus_Snn, ScalarTimesOperator) and one_minus_Snn.term is IdentityOperator:
                one_minus_Snn = one_minus_Snn.coeff
            else:
                raise AlgebraError('Inversion not implemented for general operators')

        one_minus_Snn_inv = sympyOne / one_minus_Snn

        new_S = S[:n, :n] + S[0:n, n:] * one_minus_Snn_inv * S[n:, 0: n]

        new_L = L[:n] + S[0:n, n] * one_minus_Snn_inv * L[n]
        delta_H = Im((L[:-1].adjoint() * S[:-1, n:]) * one_minus_Snn_inv * L[n, 0])

        if isinstance(delta_H, Matrix):
            delta_H = delta_H[0, 0]
        new_H = H + delta_H

        return SLH(new_S, new_L, new_H)


    def symbolic_liouvillian(self):
        from super_operator_algebra import liouvillian

        return liouvillian(self.H, self.L)

    #noinspection PyRedeclaration
    def symbolic_master_equation(self, rho=None):
        """
        Compute the symbolic Liouvillian acting on a state rho.
        If no rho is given, an OperatorSymbol is created in its place.
        This correspnds to the RHS of the master equation
        in which an average is taken over the external noise degrees of freedom.

        :param rho: A symbolic density matrix operator
        :type rho: Operator
        :return: The RHS of the master equation.
        :rtype: Operator
        """
        L, H = self.L, self.H
        if rho is None:
            rho = OperatorSymbol('rho', self.space)
        return -I * (H * rho - rho * H) + sum(Lk * rho * adjoint(Lk)
                                              - (adjoint(Lk) * Lk * rho + rho * adjoint(Lk) * Lk) / 2
                                              for Lk in L.matrix.flatten())


    #noinspection PyRedeclaration
    def symbolic_heisenberg_eom(self, X=None, noises=None):
        """
        Compute the symbolic Heisenberg equations of motion of a system operator X.
        If no X is given, an OperatorSymbol is created in its place.
        If no noises are given, this correspnds to the ensemble-averaged Heisenberg equation of motion.

        :param X: A system operator
        :type X: Operator
        :param noises: A vector of noise inputs
        :type noises: Operator
        :return: The RHS of the Heisenberg equations of motion of X.
        :rtype: Operator
        """
        L, H = self.L, self.H

        if X is None:
            X = OperatorSymbol('X', L.space | H.space)

        ret = I * (H * X - X * H) + sum(adjoint(Lk) * X * Lk \
                                        - (adjoint(Lk) * Lk * X + X * adjoint(Lk) * Lk) / 2 \
                                        for Lk in L.matrix.flatten())
        if noises is not None:
            if not isinstance(noises, Matrix):
                noises = Matrix(noises)
            LambdaT = (noises.conjugate() * noises.transpose()).transpose()
            assert noises.shape == L.shape
            S = self.S
            ret += (adjoint(noises) * S.adjoint() * (X * L - L * X)).expand()[0, 0] \
                   + ((L.adjoint() * X - X * L.adjoint()) * S * noises).expand()[0, 0]
            if len(S.space & X.space):
                comm = (S.adjoint() * X * S - X)
                ret += (comm * LambdaT).expand().trace()
        return ret

    def __iter__(self):
        return iter((self.S, self.L, self.H))

    def __len__(self):
        return 3

    def _toABCD(self, linearize):
        #TODO implement SLH._toABCD()
        pass

    def _coherent_input(self, *input_amps):
        return super(SLH, self)._coherent_input(*input_amps).toSLH()


#    def _mathematica(self):
#        return "SLH[%s, %s, %s]" % (mathematica(self.S), mathematica(self.L), mathematica(self.H))


#TODO ADD ABCD class and toABCD() methods
@check_signature
class ABCD(Circuit, Operation):
    r"""
    ABCD model class in amplitude representation.

        ``ABCD(A, B, C, D, w, space)``

    I.e. for a doubled up vector a = (a_1, ..., a_n, a_1^*, ... a_n^*)^T = double_up((a_1, ..., a_n)^T)
    and doubled up noises dA = (dA_1, ..., dA_m, dA_1^*, ..., dA_m^*)^T = double_up((dA_1, ..., dA_n)^T)
    The equation of motion for a is

    .. math::
        da = A a dt + B (dA + double_up(w) dt)

    The output field dA' is given by

    .. math::
        dA' = C a dt + D (dA + double_up(w) dt)


    :param A: Coupling matrix: internal to internal, scalar valued elements, ``shape = (2*n,2*n)``
    :type A: Matrix
    :param B: Coupling matrix external input to internal, scalar valued elements, ``shape = (2*n,2*m)``
    :type B: Matrix
    :param C: Coupling matrix internal to external output, scalar valued elements, ``shape = (2*m,2*n)``
    :type C: Matrix
    :param D: Coupling matrix external input to output, scalar valued elements, ``shape = (2*m,2*m)``
    :type D: Matrix
    :param w: Coherent input amplitude vector, **NOT DOUBLED UP**, scalar valued elements, ``shape = (m,1)``
    :type w: Matrix
    :param space: Hilbert space with exactly n local factor spaces corresponding to the n internal degrees of freedom.
    :type space: HilbertSpace
    """
    signature = Matrix, Matrix, Matrix, Matrix, Matrix, HilbertSpace

    #noinspection PyArgumentList
    def __init__(self, A, B, C, D, w, space):
        super(ABCD, self).__init__(A, B, C, D, w, space)

    @classmethod
    def create(cls, A, B, C, D, w, space):
        """
        See ABCD documentation
        """
        n2, m2 = B.shape
        if not (n2 % 2):
            raise ValueError()
        if not (m2 % 2):
            raise ValueError()
        n, m = n2 / 2, m2 / 2

        if not A.shape == (n2, n2):
            raise ValueError()

        if not C.shape == (m2, n2):
            raise ValueError()

        if not D.shape == (m2, m2):
            raise ValueError()

        if not w.shape == (m, 1):
            raise ValueError()

        if not len(space.local_factors()) == n:
            raise AlgebraError(str(space) + " != " + str(n))

        return super(ABCD, cls).create(A, B, C, D, w, space)

    @property
    def A(self):
        """Coupling matrix: internal to internal, scalar valued elements, ``shape = (2*n,2*n)``"""
        return self.operands[0]

    @property
    def B(self):
        """Coupling matrix external input to internal, scalar valued elements, ``shape = (2*n,2*m)``"""
        return self.operands[1]

    @property
    def C(self):
        """Coupling matrix internal to external output, scalar valued elements, ``shape = (2*m,2*n)``"""
        return self.operands[2]

    @property
    def D(self):
        """Coupling matrix external input to output, scalar valued elements, ``shape = (2*m,2*m)``"""
        return self.operands[3]

    @property
    def w(self):
        """Coherent input amplitude vector, **NOT DOUBLED UP**, scalar valued elements, ``shape = (m,1)``"""
        return self.operands[4]


    @property
    def _space(self):
        """
        :rtype: HilbertSpace
        """
        return self.operands[5]

    @property
    def n(self):
        """
        The number of oscillators.

        :rtype: int
        """
        return len(self.space.local_factors())

    @property
    def m(self):
        """
        The number of external fields.

        :rtype: int
        """
        return self.D.shape[0] / 2

    @property
    def _cdim(self):
        return self.m

    def _get_blocks(self, block_structure):
        return self,

    @property
    def _block_structure(self):
        return self.cdim,


    def _toABCD(self, linearize):
        return self


    def _toSLH(self):
        # TODO IMPLEMENT ABCD._toSLH()
        doubled_up_as = vstackm((Matrix([[Destroy(spc) for spc in self.space.local_factors()]]).T,
                                 Matrix([[Create(spc) for spc in self.space.local_factors()]]).T))


@check_signature
class CircuitSymbol(Circuit, Operation):
    """
    Circuit Symbol object, parametrized by an identifier and channel dimension.

        ``CircuitSymbol(identifier, cdim)``

    :type identifier: basestring
    :type cdim: int >= 0
    """
    signature = basestring, int

    def __str__(self):
        return self.identifier

    def _tex(self):
        return identifier_to_tex(self.identifier)

    @property
    def identifier(self):
        """
        The symbol identifier

        :type: str
        """
        return self.operands[0]

    @property
    def _cdim(self):
        return self.operands[1]


    def _toABCD(self, linearize):
        raise CannotConvertToABCD()

    def _toSLH(self):
        raise CannotConvertToSLH()

    def _creduce(self):
        return self

    _space = FullSpace


@singleton
class CIdentity(Circuit, Expression):
    """
    Single channel circuit identity system, the neutral element of single channel series products.
    """

    _cdim = 1

    def __str__(self):
        return "cid(1)"

    def _tex(self):
        return r"\mathbf{1}_1"

    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented

        if self.cdim == other.cdim:
            if self is other:
                return True
            try:
                return self.toSLH() == other.toSLH()
            except CannotConvertToSLH:
                return False
        return False

    def _toSLH(self):
        return SLH(Matrix([[1]]), Matrix([[0]]), 0)

    def _creduce(self):
        return self

    def _series_inverse(self):
        return self

    def _toABCD(self, linearize):
        return ABCD(zerosm((0, 0)), zerosm((0, 2)), zerosm((2, 0)), identity_matrix(2), zerosm((1, 1)), TrivialSpace)

    def _all_symbols(self):
        return {self}

    @property
    def _space(self):
        return TrivialSpace


@singleton
class CircuitZero(Circuit, Expression):
    """
    The zero circuit system, the neutral element of Concatenation. No ports, no internal dynamics.
    """
    _cdim = 0

    def __str__(self):
        return "cid(0)"

    def _tex(self):
        return r"\mathbf{1}_0"

    def __eq__(self, other):
        if self is other:
            return True
        if self.cdim == other.cdim:
            try:
                return self.toSLH() == other.toSLH()
            except CannotConvertToSLH:
                return False
        return False

    def _toSLH(self):
        return SLH(Matrix([[]]), Matrix([[]]), 0)

    def _toABCD(self, linearize):
        return ABCD(zerosm((0, 0)), zerosm((0, 0)), zerosm((0, 0)), zerosm((0, 0)), zerosm((0, 0)), TrivialSpace)

    def _creduce(self):
        return self

    def _all_symbols(self):
        return {}

    @property
    def _space(self):
        return TrivialSpace


cid_1 = CIdentity


def circuit_identity(n):
    """
    Return the circuit identity for n channels.

    :param n: The channel dimension
    :type n: int
    :return: n-channel identity circuit
    :rtype: Circuit
    """
    if n <= 0:
        return CircuitZero
    if n == 1:
        return cid_1
    return Concatenation(*((cid_1,) * n))


cid = circuit_identity



#noinspection PyRedeclaration,PyTypeChecker
def get_common_block_structure(lhs_bs, rhs_bs):
    """
    For two block structures ``aa = (a1, a2, ..., an)``, ``bb = (b1, b2, ..., bm)``
    generate the maximal common block structure so that every block from aa and bb
    is contained in exactly one block of the resulting structure.
    This is useful for determining how to apply the distributive law when feeding
    two concatenated Circuit objects into each other.

    Examples:
        ``(1, 1, 1), (2, 1) -> (2, 1)``
        ``(1, 1, 2, 1), (2, 1, 2) -> (2, 3)``

    :param lhs_bs: first block structure
    :type lhs_bs: tuple
    :param rhs_bs: second block structure
    :type rhs_bs: tuple

    """

    # for convenience the arguments may also be Circuit objects
    if isinstance(lhs_bs, Circuit):
        lhs_bs = lhs_bs.block_structure
    if isinstance(rhs_bs, Circuit):
        rhs_bs = rhs_bs.block_structure

    if sum(lhs_bs) != sum(rhs_bs):
        raise AlgebraError('Blockstructures have different total channel numbers.')

    if len(lhs_bs) == len(rhs_bs) == 0:
        return ()

    i = j = 1
    lsum = 0
    while True:
        lsum = sum(lhs_bs[:i])
        rsum = sum(rhs_bs[:j])
        if lsum < rsum:
            i += 1
        elif rsum < lsum:
            j += 1
        else:
            break

    return (lsum, ) + get_common_block_structure(lhs_bs[i:], rhs_bs[j:])


def check_cdims_mtd(dcls, clsmtd, cls, *ops):
    """
    Check that all operands (`ops`) have equal channel dimension.
    """
    if not len({o.cdim for o in ops}) == 1:
        raise ValueError("Not all operands have the same cdim:" + str(ops))
    return clsmtd(cls, *ops)


check_cdims = preprocess_create_with(check_cdims_mtd)


@assoc
@filter_neutral
@check_cdims
@match_replace_binary
@filter_neutral
@check_signature_assoc
class SeriesProduct(Circuit, Operation):
    """
    The series product circuit operation. It can be applied to any sequence of circuit objects that have equal channel dimension.

        ``SeriesProduct(*operands)``

    :param operands: Circuits in feedforward configuration.

    """
    signature = Circuit,
    _binary_rules = [
    ]


    @singleton
    class neutral_element(object):
        """
        Generic neutral element checker of the ``SeriesProduct``, it works for any channel dimension.
        """

        def __eq__(self, other):
            #            print("neutral?", other)
            return self is other or other == cid(other.cdim)

        def __ne__(self, other):
            return not (self == other)

    @property
    def _cdim(self):
        return self.operands[0].cdim

    def _toSLH(self):
        return reduce(lambda a, b: a.toSLH().series_with_slh(b.toSLH()), self.operands)

    def _creduce(self):
        return SeriesProduct.create(*[op.creduce() for op in self.operands])

    def _series_inverse(self):
        return SeriesProduct.create(*[o.series_inverse() for o in reversed(self.operands)])

    def _tex(self):
        ret = " \lhd ".join("{{{}}}".format(o.tex()) if not isinstance(o, Concatenation)
                            else r"\left({}\right)".format(o.tex()) for o in self.operands)
        #        print(ret)
        return ret

    def __str__(self):
        return " << ".join(str(o) if not isinstance(o, Concatenation)
                           else r"({!s})".format(o) for o in self.operands)

    def _toABCD(self, linearize):
        # TODO implement SeriesProduct._toABCD()
        pass

    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)


@assoc
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_assoc
class Concatenation(Circuit, Operation):
    """
    The concatenation product circuit operation. It can be applied to any sequence of circuit objects.

        ``Concatenation(*operands)``

    :param Circuit operands: Circuits in parallel configuration.
    """

    signature = Circuit,

    neutral_element = CircuitZero

    _binary_rules = []

    def _tex(self):
        ops_strs = []
        id_count = 0
        for o in self.operands:
            if o == CIdentity:
                id_count += 1
            else:
                if id_count > 0:
                    ops_strs += [r"\mathbf{{1}}_{{{}}}".format(id_count)]
                    id_count = 0
                ops_strs += [tex(o) if not isinstance(o, SeriesProduct) else "({})".format(o.tex())]
        if id_count > 0:
            ops_strs += [r"\mathbf{{1}}_{{{}}}".format(id_count)]
        return r" \boxplus ".join(ops_strs)


    def __str__(self):
        ops_strs = []
        id_count = 0
        for o in self.operands:
            if o == CIdentity:
                id_count += 1
            else:
                if id_count > 0:
                    ops_strs += ["cid({})".format(id_count)]
                    id_count = 0
                ops_strs += [str(o) if not isinstance(o, SeriesProduct) else "({!s})".format(o)]
        if id_count > 0:
            ops_strs += ["cid({})".format(id_count)]
        return " + ".join(ops_strs)


    @property
    def _cdim(self):
        return sum((circuit.cdim for circuit in self.operands))

    def _toSLH(self):
        return reduce(lambda a, b: a.toSLH().concatenate_slh(b.toSLH()), self.operands)


    def _creduce(self):
        return Concatenation.create(*[op.creduce() for op in self.operands])

    @property
    def _block_structure(self):
        return sum((circuit.block_structure for circuit in self.operands), ())


    def _get_blocks(self, block_structure):

        blocks = []
        block_iter = iter(sum((op.get_blocks() for op in self.operands), ()))
        cbo = []
        current_length = 0
        for bl in block_structure:
            while current_length < bl:
                next_op = next(block_iter)
                cbo.append(next_op)
                current_length += next_op.cdim
            if current_length != bl:
                raise IncompatibleBlockStructures('requested blocks according to incompatible block_structure')
            blocks.append(Concatenation.create(*cbo))
            cbo = []
            current_length = 0
        return tuple(blocks)

    def _series_inverse(self):
        return Concatenation.create(*[o.series_inverse() for o in self.operands])

    def _feedback(self, out_index, in_index):

        n = self.cdim

        if out_index == n - 1 and in_index == n - 1:
            return Concatenation.create(*(self.operands[:-1] + (self.operands[-1].feedback(),)))

        in_index_in_block, in_block = self.index_in_block(in_index)
        out_index_in_block, out_block = self.index_in_block(out_index)

        blocks = self.get_blocks()

        if in_block == out_block:
            return Concatenation.create(*blocks[:out_block]) \
                   + blocks[out_block].feedback(out_index=out_index_in_block, in_index=in_index_in_block) \
                   + Concatenation.create(*blocks[out_block + 1:])
        ### no 'real' feedback loop, just an effective series
        #partition all blocks into just two


        if in_block < out_block:
            b1 = Concatenation.create(*blocks[:out_block])
            b2 = Concatenation.create(*blocks[out_block:])

            return (b1 + circuit_identity(b2.cdim - 1)) \
                   << map_signals_circuit({out_index - 1: in_index}, n - 1) \
                   << (circuit_identity(b1.cdim - 1) + b2)
        else:
            b1 = Concatenation.create(*blocks[:in_block])
            b2 = Concatenation.create(*blocks[in_block:])

            return (circuit_identity(b1.cdim - 1) + b2) \
                   << map_signals_circuit({out_index: in_index - 1}, n - 1) \
                   << (b1 + circuit_identity(b2.cdim - 1))

    def _toABCD(self, linearize):
        # TODO implement Concatenation._toABCD()
        pass

    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)


class CannotFactorize(Exception):
    pass


@check_signature
class CPermutation(Circuit, Operation):
    r"""
    The channel permuting circuit. This circuit expression is only a rearrangement of input and output fields.
    A channel permutation is given as a tuple of image points. Permutations are usually represented as

    A permutation :math:`\sigma \in \Sigma_n` of :math:`n` elements  is often represented in the following form

    .. math::
         \begin{pmatrix}
                1           &       2   & \dots &   n       \\
                \sigma(1)   & \sigma(2) & \dots & \sigma(n)
        \end{pmatrix},

    but obviously it is fully sufficient to specify the tuple of images :math:`(\sigma(1), \sigma(2), \dots, \sigma(n))`.
    We thus parametrize our permutation circuits only in terms of the image tuple.
    Moreover, we will be working with *zero-based indices*!

    A channel permutation circuit for a given permutation (represented as a python tuple of image indices)
    scatters the :math:`j`-th input field to the :math:`\sigma(j)`-th output field.

    It is instantiated as

        ``CPermutation(permutation)``

    :param permutation: Channel permutation image tuple.
    :type permutation: tuple
    """
    signature = tuple,

    #noinspection PyDocstring
    @classmethod
    def create(cls, permutation):
        """
        See CPermutation docs.
        """
        if not check_permutation(permutation):
            raise BadPermutationError(str(permutation))
        if list(permutation) == list(range(len(permutation))):
            return cid(len(permutation))
        return super(CPermutation, cls).create(permutation)

    _block_perms = None

    @property
    def block_perms(self):
        """
        If the circuit is reducible into permutations within subranges of the full range of channels,
        this yields a tuple with the internal permutations for each such block.

        :type: tuple
        """
        if not self._block_perms:
            self._block_perms = permutation_to_block_permutations(self.permutation)
        return self._block_perms


    @property
    def permutation(self):
        """
        The permutation image tuple.

        :type: tuple
        """
        return self.operands[0]


    def _toSLH(self):
        return SLH(permutation_matrix(self.permutation), zerosm((self.cdim, 1)), 0)

    def _toABCD(self, linearize):
        return self.toSLH().toABCD()

    def __str__(self):
        return "P_sigma{!r}".format(self.permutation)

    @property
    def _cdim(self):
        return len(self.permutation)

    def _creduce(self):
        return self

    def _tex(self):
        return "\mathbf{{P}}_\sigma \\begin{{pmatrix}} {} \\\ {} \\end{{pmatrix}}".format(
            " & ".join(map(str, range(self.cdim))), " & ".join(map(str, self.permutation)))

    def series_with_permutation(self, other):
        """
        Compute the series product with another channel permutation circuit.

        :type other: CPermutation
        :return: The composite permutation circuit (could also be the identity circuit for n channels)
        :rtype: Circuit
        """
        combined_permutation = tuple([self.permutation[p] for p in other.permutation])
        return CPermutation.create(combined_permutation)


    def _series_inverse(self):
        return CPermutation(invert_permutation(self.permutation))

    @property
    def _block_structure(self):
        return tuple(map(len, self.block_perms))

    def _get_blocks(self, block_structure):

        block_perms = []

        if block_structure == self.block_structure:
            return tuple(map(CPermutation.create, self.block_perms))

        if len(block_structure) > len(self.block_perms):
            raise Exception
        if sum(block_structure) != self.cdim:
            raise Exception
        current_perm = []
        block_perm_iter = iter(self.block_perms)
        for l in block_structure:
            while len(current_perm) < l:
                offset = len(current_perm)
                current_perm += [p + offset for p in next(block_perm_iter)]

            if len(current_perm) != l:
                # print(block_structure, self.block_perms, block_perms)
                raise Exception

            block_perms.append(tuple(current_perm))
            current_perm = []
        return tuple(map(CPermutation.create, block_perms))


    def _factorize_for_rhs(self, rhs):
        """
        Factorize a channel permutation circuit according the block structure of the upstream circuit.
        This allows to move as much of the permutation as possible *around* a reducible circuit upstream.
        It basically decomposes

            ``permutation << rhs --> permutation' << rhs' << residual'``

        where rhs' is just a block permutated version of rhs and residual'
        is the maximal part of the permutation that one may move around rhs.

        :param rhs: An upstream circuit object
        :type rhs: Circuit
        :return: new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit
        :rtype: tuple
        :raise: BadPermutationError
        """
        block_structure = rhs.block_structure

        block_perm, perms_within_blocks = block_perm_and_perms_within_blocks(self.permutation, block_structure)
        fblockp = full_block_perm(block_perm, block_structure)

        if not sorted(fblockp) == list(range(self.cdim)):
            raise BadPermutationError()

        new_rhs_circuit = CPermutation.create(fblockp)
        within_blocks = [CPermutation.create(within_block) for within_block in perms_within_blocks]
        within_perm_circuit = sum(within_blocks, cid(0))
        rhs_blocks = rhs.get_blocks(block_structure)

        permuted_rhs_circuit = Concatenation.create(*[SeriesProduct.create(within_blocks[p], rhs_blocks[p]) \
                                                      for p in invert_permutation(block_perm)])

        new_lhs_circuit = self << within_perm_circuit.series_inverse() << new_rhs_circuit.series_inverse()

        return new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit


    def _feedback(self, out_index, in_index):
        n = self.cdim
        new_perm_circuit = map_signals_circuit({out_index: (n - 1)}, n) << self << map_signals_circuit(
            {(n - 1): in_index}, n)
        if new_perm_circuit == circuit_identity(n):
            return circuit_identity(n - 1)
        new_perm = list(new_perm_circuit.permutation)
        n_inv = new_perm.index(n - 1)
        new_perm[n_inv] = new_perm[n - 1]

        return CPermutation.create(tuple(new_perm[:-1]))


    def _factor_rhs(self, in_index):
        """
        With::

            n           := self.cdim
            in_im       := self.permutation[in_index]
            m_{k->l}    := map_signals_circuit({k:l}, n)

        solve the equation (I) containing ``self``::

            self << m_{(n-1) -> in_index} == m_{(n-1) -> in_im} << (red_self + cid(1))          (I)

        for the (n-1) channel CPermutation ``red_self``.
        Return in_im, red_self.

        This is useful when ``self`` is the RHS in a SeriesProduct Object that is within a Feedback loop
        as it allows to extract the feedback channel from the permutation and moving the
        remaining part of the permutation (``red_self``) outside of the feedback loop.

        :param int in_index: The index for which to factor.
        """
        n = self.cdim
        if not (0 <= in_index < n):
            raise Exception
        in_im = self.permutation[in_index]
        # (I) is equivalent to
        #       m_{in_im -> (n-1)} <<  self << m_{(n-1) -> in_index} == (red_self + cid(1))     (I')
        red_self_plus_cid1 = map_signals_circuit({in_im: (n - 1)}, n) << self << map_signals_circuit(
            {(n - 1): in_index}, n)
        if isinstance(red_self_plus_cid1, CPermutation):

            #make sure we can factor
            #noinspection PyUnresolvedReferences
            assert red_self_plus_cid1.permutation[(n - 1)] == (n - 1)

            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])

            return in_im, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_index ->  in_im}

            return in_im, circuit_identity(n - 1)

    def _factor_lhs(self, out_index):
        """
        With::

            n           := self.cdim
            out_inv     := invert_permutation(self.permutation)[out_index]
            m_{k->l}    := map_signals_circuit({k:l}, n)

        solve the equation (I) containing ``self``::

            m_{out_index -> (n-1)} << self == (red_self + cid(1)) << m_{out_inv -> (n-1)}           (I)

        for the (n-1) channel CPermutation ``red_self``.
        Return out_inv, red_self.

        This is useful when 'self' is the LHS in a SeriesProduct Object that is within a Feedback loop
        as it allows to extract the feedback channel from the permutation and moving the
        remaining part of the permutation (``red_self``) outside of the feedback loop.

        :param out_index: The index for which to factor
        """
        n = self.cdim
        if not (0 <= out_index < n):
            print(self, out_index)
            raise Exception
        out_inv = self.permutation.index(out_index)

        # (I) is equivalent to
        #       m_{out_index -> (n-1)} <<  self << m_{(n-1) -> out_inv} == (red_self + cid(1))     (I')

        red_self_plus_cid1 = map_signals_circuit({out_index: (n - 1)}, n) << self << map_signals_circuit(
            {(n - 1): out_inv}, n)

        if isinstance(red_self_plus_cid1, CPermutation):

            #make sure we can factor
            assert red_self_plus_cid1.permutation[(n - 1)] == (n - 1)

            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])

            return out_inv, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_index ->  in_im}

            return out_inv, circuit_identity(n - 1)

    @property
    def _space(self):
        return TrivialSpace


def P_sigma(*permutation):
    """
    Create a channel permutation circuit for the given index image values.
    :param permutation: image points
    :type permutation: int
    :return: CPermutation.create(permutation)
    :rtype: Circuit
    """
    return CPermutation.create(permutation)


def extract_signal(k, n):
    """
    Create a permutation that maps the k-th (zero-based) element to the last element,
    while preserving the relative order of all other elements.
    :param k: The index to extract
    :type k: int
    :param n: The total number of elements
    :type n: int
    :return: Permutation image tuple
    :rtype: tuple
    """
    return tuple(range(k) + [n - 1] + range(k, n - 1))


def extract_signal_circuit(k, cdim):
    """
    Create a channel permutation circuit that maps the k-th (zero-based) input to the last output,
    while preserving the relative order of all other channels.
    :param k: Extracted channel index
    :type k: int
    :param cdim: The channel dimension
    :type cdim: int
    :return: Permutation circuit
    :rtype: Circuit
"""
    return CPermutation.create(extract_signal(k, cdim))


def map_signals(mapping, n):
    """
    For a given {input:output} mapping in form of a dictionary,
    generate the permutation that achieves the specified mapping
    while leaving the relative order of all non-specified elements intact.
    :param mapping: Input-output mapping of indices (zero-based) {in1:out1, in2:out2,...}
    :type mapping: dict
    :param n: total number of elements
    :type n: int
    :return: Signal mapping permutation image tuple
    :rtype: tuple
    :raise: ValueError
    """
    free_values = list(range(n))

    for v in mapping.values():
        if v >= n:
            raise ValueError('the mapping cannot take on values larger than cdim - 1')
        free_values.remove(v)
    for k in mapping:
        if k >= n:
            raise ValueError('the mapping cannot map keys larger than cdim - 1')
    # sorted(set(range(n)).difference(set(mapping.values())))
    permutation = []
    # print(free_values, mapping, n)
    for k in range(n):
        if k in mapping:
            permutation.append(mapping[k])
        else:
            permutation.append(free_values.pop(0))
    # print(permutation)
    return tuple(permutation)


def map_signals_circuit(mapping, n):
    """
    For a given {input:output} mapping in form of a dictionary,
    generate the channel permutating circuit that achieves the specified mapping
    while leaving the relative order of all non-specified channels intact.
    :param mapping: Input-output mapping of indices (zero-based) {in1:out1, in2:out2,...}
    :type mapping: dict
    :param n: total number of elements
    :type n: int
    :return: Signal mapping permutation image tuple
    :rtype: Circuit
    """
    return CPermutation.create(map_signals(mapping, n))


def pad_with_identity(circuit, k, n):
    """
    Pad a circuit by 'inserting' an n-channel identity circuit at index k.
    I.e., a circuit of channel dimension N is extended to one of channel dimension N+n, where the channels
    k, k+1, ...k+n-1, just pass through the system unaffected.
    E.g. let A, B be two single channel systems

        >>> A = CircuitSymbol('A', 1)
        >>> B = CircuitSymbol('B', 1)
        >>> pad_with_identity(A+B, 1, 2)
            (A + cid(2) + B)

    This method can also be applied to irreducible systems, but in that case the result can not be decomposed as nicely.

    :type circuit: Circuit
    :param k: The index at which to insert the circuit
    :type k: int
    :param n: The number of channels to pass through
    :type n: int
    :return: An extended circuit that passes through the channels k, k+1, ..., k+n-1
    :rtype: Circuit
    """
    circuit_n = circuit.cdim
    combined_circuit = circuit + circuit_identity(n)
    permutation = range(k) + range(circuit_n, circuit_n + n) + range(k, circuit_n)
    return CPermutation.create(invert_permutation(permutation)) << combined_circuit << CPermutation.create(permutation)


@match_replace
@check_signature
class Feedback(Circuit, Operation):
    """
    The circuit feedback operation applied to a circuit of channel dimension > 1
    and an from an output port index to an input port index.

        ``Feedback(circuit, out_index, in_index)``

    :param circuit: The circuit that undergoes self-feedback
    :type circuit: Circuit
    :param out_index: The output port index.
    :type out_index: int
    :param in_index: The input port index.
    :type in_index: int
    """
    delegate_to_method = (Concatenation, SLH, CPermutation)
    signature = Circuit, int, int

    _rules = []

    @property
    def operand(self):
        """
        The circuit that undergoes feedback
        :rtype: Circuit
        """
        return self._operands[0]

    @property
    def out_in_pair(self):
        """
        Zero-based feedback port indices (out_index, in_index)
        :rtype: tuple
        """
        return self._operands[1:]

    @property
    def _cdim(self):
        return self.operand.cdim - 1

    #noinspection PyUnresolvedReferences
    @classmethod
    def create(cls, circuit, out_index, in_index):
        """
        See :py:class:Feedback: documentation.
        """
        if not isinstance(circuit, Circuit):
            raise ValueError()

        n = circuit.cdim
        if not n:
            raise ValueError()

        if n == 1:
            raise ValueError()

        if isinstance(circuit, cls.delegate_to_method):
            return circuit._feedback(out_index, in_index)

        return super(Feedback, cls).create(circuit, out_index, in_index)


    def _toSLH(self):
        return self.operand.toSLH().feedback(*self.out_in_pair)

    def _toABCD(self):
        # TODO implement Feedback._toABCD()
        raise NotImplementedError(self.__class__)

    def _creduce(self):
        return self.operand.creduce().feedback(*self.out_in_pair)


    #    def substitute(self, var_map):
    #        op = substitute(self.operand, var_map)
    #        return op.feedback(*self.out_in_pair)

    def __str__(self):
        if self.out_in_pair == (self.operand.cdim - 1, self.operand.cdim - 1):
            return "FB(%s)" % self.operand
        o, i = self.out_in_pair
        return "FB(%s, %d, %d)" % (self.operand, o, i)

    def _tex(self):
        o, i = self.out_in_pair
        if self.out_in_pair == (self.cdim - 1, self.cdim - 1):
            return "\left\lfloor%s\\right\\rfloor" % tex(self.operand)
        return "\left\lfloor%s\\right\\rfloor_{%d\\to%d}" % (tex(self.operand), o, i)


    def _series_inverse(self):
        return Feedback.create(self.operand.series_inverse(), *reversed(self.out_in_pair))

    @property
    def _space(self):
        return self.operand.space


#noinspection PyRedeclaration
def FB(circuit, out_index=None, in_index=None):
    """
    Wrapper for :py:class:Feedback: but with additional default values.

        ``FB(circuit, out_index = None, in_index = None)``

    :param circuit: The circuit that undergoes self-feedback
    :type circuit: Circuit
    :param out_index: The output port index, default = None --> last port
    :type out_index: int
    :param in_index: The input port index, default = None --> last port
    :type in_index: int
    :return: The circuit with applied feedback operation.
    :rtype: Circuit
    """
    if out_index is None:
        out_index = circuit.cdim - 1
    if in_index is None:
        in_index = circuit.cdim - 1
    return Feedback.create(circuit, out_index, in_index)


@check_signature
class SeriesInverse(Circuit, Operation):
    """
    Symbolic series product inversion operation.

        ``SeriesInverse(circuit)``

    One generally has

        >>> SeriesInverse(circuit) << circuit == cid(circuit.cdim)
            True

    and

        >>> circuit << SeriesInverse(circuit) == cid(circuit.cdim)
            True

    :param Circuit circuit: The circuit system to invert.
    """
    signature = Circuit,

    delegate_to_method = (SeriesProduct, Concatenation, Feedback, SLH, CPermutation, CIdentity.__class__)

    @property
    def operand(self):
        """
        The un-inverted circuit

        :rtype: Circuit
        """
        return self.operands[0]


    @classmethod
    def create(cls, circuit):
        """
        See documentation for :py:class:`SeriesProduct`
        """
        if isinstance(circuit, SeriesInverse):
            return circuit.operand

        elif isinstance(circuit, cls.delegate_to_method):
            #noinspection PyUnresolvedReferences
            return circuit._series_inverse()

        return super(SeriesInverse, cls).create(circuit)

    @property
    def _cdim(self):
        return self.operand.cdim


    def _toSLH(self):
        return self.operand.toSLH().series_inverse()

    def _toABCD(self):
        raise AlgebraError("SeriesInverse not well-defined in ABCD model context")

    def _creduce(self):
        return self.operand.creduce().series_inverse()

    def _substitute(self, var_map):
        return substitute(self, var_map).series_inverse()

    @property
    def _space(self):
        return self.operand.space

    def __str__(self):
        return "[{!s}]^(-1)".format(self.operand)

    def _tex(self):
        return r"\left[ {} \right]^{{\lhd -1}}".format(tex(self.operand))


def _tensor_decompose_series(lhs, rhs):
    """
    Simplification method for lhs << rhs
    Decompose a series product of two reducible circuits with compatible block structures into
    a concatenation of individual series products between subblocks.
    This method raises CannotSimplify when rhs is a CPermutation in order not to conflict with other _rules.
    :type lhs: Circuit
    :type rhs: Circuit
    :return: The combined reducible circuit
    :rtype: Circuit
    :raise: CannotSimplify
    """
    if isinstance(rhs, CPermutation):
        raise CannotSimplify()
    res_struct = get_common_block_structure(
        lhs.block_structure,
        rhs.block_structure
        )
    if len(res_struct) > 1:
        blocks, oblocks = (
            lhs.get_blocks(res_struct),
            rhs.get_blocks(res_struct))
        parallel_series = [SeriesProduct.create(lb, rb)
                           for (lb, rb) in zip(blocks, oblocks)]
        return Concatenation.create(*parallel_series)
    raise CannotSimplify()


def _factor_permutation_for_blocks(cperm, rhs):
    """
    Simplification method for cperm << rhs.
    Decompose a series product of a channel permutation and a reducible circuit with appropriate block structure
    by decomposing the permutation into a permutation within each block of rhs and a block permutation and a residual part.
    This allows for achieving something close to a normal form for circuit expression.
    :type cperm: CPermutation
    :type rhs: Circuit
    :rtype: Circuit
    :raise: CannotSimplify
    """
    rbs = rhs.block_structure
    if rhs == cid(rhs.cdim):
        return cperm
    if len(rbs) > 1:
        residual_lhs, transformed_rhs, carried_through_lhs = cperm._factorize_for_rhs(rhs)
        if residual_lhs == cperm:
            raise CannotSimplify()
        return SeriesProduct.create(residual_lhs, transformed_rhs, carried_through_lhs)
    raise CannotSimplify()


def _pull_out_perm_lhs(lhs, rest, out_index, in_index):
    """
    Pull out a permutation from the Feedback of a SeriesProduct with itself.

    :param lhs: The permutation circuit
    :type lhs: CPermutation
    :param rest: The other SeriesProduct operands
    :type rest: OperandsTuple
    :param out_index: The feedback output port index
    :type out_index: int
    :param in_index: The feedback input port index
    :type in_index: int
    :return: The simplified circuit
    :rtype: Circuit
    """
    out_inv, lhs_red = lhs._factor_lhs(out_index)
    return lhs_red << Feedback.create(SeriesProduct.create(*rest), out_inv, in_index)


def _pull_out_unaffected_blocks_lhs(lhs, rest, out_index, in_index):
    """
    In a self-Feedback of a series product, where the left-most operand is reducible,
    pull all non-trivial blocks outside of the feedback.

   :param lhs: The reducible circuit
   :type lhs: Circuit
   :param rest: The other SeriesProduct operands
   :type rest: OperandsTuple
   :param out_index: The feedback output port index
   :type out_index: int
   :param in_index: The feedback input port index
   :type in_index: int
   :return: The simplified circuit
   :rtype: Circuit
   """

    _, block_index = lhs.index_in_block(out_index)

    bs = lhs.block_structure

    nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
    before, block, after = lhs.get_blocks((nbefore, nblock, nafter))

    if before != cid(nbefore) or after != cid(nafter):
        outer_lhs = before + cid(nblock - 1) + after
        inner_lhs = cid(nbefore) + block + cid(nafter)
        return outer_lhs << Feedback.create(SeriesProduct.create(inner_lhs, *rest), out_index, in_index)
    elif block == cid(nblock):
        outer_lhs = before + cid(nblock - 1) + after
        return outer_lhs << Feedback.create(SeriesProduct.create(*rest), out_index, in_index)
    raise CannotSimplify()


#noinspection PyDocstring
def _pull_out_perm_rhs(rest, rhs, out_index, in_index):
    """
    Similar to :py:func:_pull_out_perm_lhs: but on the RHS of a series product self-feedback.
    """
    in_im, rhs_red = rhs._factor_rhs(in_index)
    return Feedback.create(SeriesProduct.create(*rest), out_index, in_im) << rhs_red


def _pull_out_unaffected_blocks_rhs(rest, rhs, out_index, in_index):
    """
    Similar to :py:func:_pull_out_unaffected_blocks_lhs: but on the RHS of a series product self-feedback.
    """
    _, block_index = rhs.index_in_block(in_index)
    bs = rhs.block_structure
    nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
    before, block, after = rhs.get_blocks((nbefore, nblock, nafter))
    if before != cid(nbefore) or after != cid(nafter):
        outer_rhs = before + cid(nblock - 1) + after
        inner_rhs = cid(nbefore) + block + cid(nafter)
        return Feedback.create(SeriesProduct.create(*(rest + (inner_rhs,))), out_index, in_index) << outer_rhs
    elif block == cid(nblock):
        outer_rhs = before + cid(nblock - 1) + after
        return Feedback.create(SeriesProduct.create(*rest), out_index, in_index) << outer_rhs
    raise CannotSimplify()


#noinspection PyDocstring
def _series_feedback(series, out_index, in_index):
    """
    Invert a series self-feedback twice to get rid of unnecessary permutations.
    """
    series_s = series.series_inverse().series_inverse()
    if series_s == series:
        raise CannotSimplify()
    return series_s.feedback(out_index, in_index)


A_CPermutation = wc("A", head=CPermutation)
B_CPermutation = wc("B", head=CPermutation)
C_CPermutation = wc("C", head=CPermutation)
D_CPermutation = wc("D", head=CPermutation)

A_Concatenation = wc("A", head=Concatenation)
B_Concatenation = wc("B", head=Concatenation)

A_SeriesProduct = wc("A", head=SeriesProduct)

A_Circuit = wc("A", head=Circuit)
B_Circuit = wc("B", head=Circuit)
C_Circuit = wc("C", head=Circuit)

A__Circuit = wc("A__", head=Circuit)
B__Circuit = wc("B__", head=Circuit)
C__Circuit = wc("C__", head=Circuit)

A_SLH = wc("A", head=SLH)
B_SLH = wc("B", head=SLH)

A_ABCD = wc("A", head=ABCD)
B_ABCD = wc("B", head=ABCD)

j_int = wc("j", head=int)
k_int = wc("k", head=int)

SeriesProduct._binary_rules += [
    ((A_CPermutation, B_CPermutation), lambda A, B: A.series_with_permutation(B)),
    ((A_SLH, B_SLH), lambda A, B: A.series_with_slh(B)),
    ((A_ABCD, B_ABCD), lambda A, B: A.series_with_abcd(B)),
    ((A_Circuit, B_Circuit), lambda A, B: _tensor_decompose_series(A, B)),
    ((A_CPermutation, B_Circuit), lambda A, B: _factor_permutation_for_blocks(A, B))
]

Concatenation._binary_rules += [
    ((A_SLH, B_SLH), lambda A, B: A.concatenate_slh(B)),
    ((A_ABCD, B_ABCD), lambda A, B: A.concatenate_abcd(B)),
    ((A_CPermutation, B_CPermutation),
     lambda A, B: CPermutation.create(concatenate_permutations(A.operands[0], B.operands[0]))),
    ((A_CPermutation, CIdentity), lambda A: CPermutation.create(concatenate_permutations(A.operands[0], (0,)))),
    ((CIdentity, B_CPermutation ), lambda B: CPermutation.create(concatenate_permutations((0,), B.operands[0]))),
    ((SeriesProduct(A__Circuit, B_CPermutation), SeriesProduct(C__Circuit, D_CPermutation)),
     lambda A, B, C, D: (SeriesProduct.create(*A) + SeriesProduct.create(*C)) << (B + D)),
    ((SeriesProduct(A__Circuit, B_CPermutation), C_Circuit),
     lambda A, B, C: (SeriesProduct.create(*A) + C) << (B + cid(C.cdim))),
    ((A_Circuit, SeriesProduct(B__Circuit, C_CPermutation)),
     lambda A, B, C: (A + SeriesProduct.create(*B)) << (cid(A.cdim) + C)),
]

Feedback._rules += [
    ((A_SeriesProduct, j_int, k_int), lambda A, j, k: _series_feedback(A, j, k)),
    ((SeriesProduct(A_CPermutation, B__Circuit), j_int, k_int ), lambda A, B, j, k: _pull_out_perm_lhs(A, B, j, k)),
    ((SeriesProduct(A_Concatenation, B__Circuit), j_int, k_int ),
     lambda A, B, j, k: _pull_out_unaffected_blocks_lhs(A, B, j, k)),
    ((SeriesProduct(A__Circuit, B_CPermutation), j_int, k_int ), lambda A, B, j, k: _pull_out_perm_rhs(A, B, j, k)),
    ((SeriesProduct(A__Circuit, B_Concatenation), j_int, k_int ),
     lambda A, B, j, k: _pull_out_unaffected_blocks_rhs(A, B, j, k)),
]


# def getABCD(slh, double_up=True):
#     """
#     Return the matrices necessary to carry out a semi-classical simulation
#     of the SLH system driven by some dynamic inputs.
#
#     Params
#     ------
#     :param slh: SLH object to linearize
#     :param double_up: Whether the matrices should be returned for the more general, doubled up case
#     :rtype : tuple
#
#     Returns
#     -------
#     A tuple (A, B, C, D, a, c)
#
#     A: coupling of modes to each other
#     B: coupling of external input fields to modes
#     C: coupling of internal modes to output
#     D: coupling of external input fields to output fields
#     a: constant coherent drive term in dx_t
#     c: constant coherent drive term in dA'_t
#
#     The linearized QSDE is then:
#     dx_t = (A * x_t + a)dt + B * dA_t
#     dA'_t = (C * x_t + c) dt + D * dA_t
#
#     If double_up is False x_t corresponds only to the vector of annihilation operators.
#     If it is true, it is the doubled up form, i.e., first all annihilation operators, then all creation operators.
#
#     Here A * b is a matrix product, whereas a (*) b is an element-wise
#     product of two vectors.
#     """
#
#     # the different degrees of freedom
#     modes = sorted(slh.space.local_factors())
#
#     # various dimensions
#     ncav = len(modes)
#     cdim = slh.cdim
#
#     def _ascomplex(o):
#         try:
#             return complex(o.coeff) if (isinstance(o, ScalarTimesOperator) and o.term is IdentityOperator) else o
#         except:
#             return o
#
#
#     if double_up:
#         # initialize the matrices
#         A = np_zeros((2*ncav, 2*ncav), dtype=object)
#         B = np_zeros((2*ncav, 2*cdim), dtype=object)
#         C = np_zeros((2*cdim, 2*ncav), dtype=object)
#
#         D_ = slh.S.matrix.element_wise(_ascomplex)
#         D = block_matrix(D_, zerosm(D_.shape), zerosm(D_.shape), D_.conjugate()).matrix
#         a = np_zeros(ncav, dtype=object)
#         c = np_zeros(cdim, dtype=object)
#     else:
#         # initialize the matrices
#         A = np_zeros((ncav, ncav), dtype=object)
#         B = np_zeros((ncav, cdim), dtype=object)
#         C = np_zeros((cdim, ncav), dtype=object)
#
#         D = slh.S.matrix.element_wise(_ascomplex).matrix
#         a = np_zeros(ncav, dtype=object)
#         c = np_zeros(cdim, dtype=object)
#
#     # make symbols for the external field modes
#     noises = [OperatorSymbol('b_{{{}}}'.format(n), "ext({})".format(n)) for n in range(cdim)]
#
#     # compute the QSDEs for the internal operators
#     eoms = [slh.symbolic_heisenberg_eom(Destroy(s), noises=noises).expand().simplify_scalar() for s in modes]
#
#     # use the coefficients to generate A, B matrices
#     for jj, sjj in enumerate(modes):
#         coeffsjj = get_coeffs(eoms[jj])
#         for kk, skk in enumerate(modes):
#             A[jj, kk] = coeffsjj[Destroy(skk)]
#             if double_up:
#                 A[jj, kk+ncav] = coeffsjj[Create(skk)]
#
#         for kk, dAkk in enumerate(noises):
#             B[jj, kk] = coeffsjj[dAkk]
#             if double_up:
#                 B[jj, kk+cdim] = coeffsjj[dAkk.dag()]
#         a[jj] = coeffsjj[IdentityOperator]
#
#     # use the coefficients in the L vector to generate the C, D
#     # matrices
#     for jj, Ljj in enumerate(slh.L.matrix[:, 0]):
#         coeffsjj = get_coeffs(Ljj)
#         for kk, skk in enumerate(modes):
#             C[jj, kk] = coeffsjj[Destroy(skk)]
#             if double_up:
#                 C[jj, kk+cdim] = coeffsjj[Create(skk)]
#         c[jj] = coeffsjj[IdentityOperator]
#
#     def _extend_double_up(mat):
#         m, n = mat.shape
#         mat[m//2:, :n//2] = mat[:m//2, n//2:].conjugate()
#         mat[m // 2:, n // 2:] = mat[:m // 2, :n // 2].conjugate()
#
#     if double_up:
#
#         _extend_double_up(A)
#         _extend_double_up(B)
#         _extend_double_up(C)
#         _extend_double_up(D)
#         a[ncav:] = a[:ncav].conjugate()
#         c[cdim:] = c[:cdim].conjugate()
#
#     return map(Matrix, (A, B, C, D, a, c))


# noinspection PyPep8Naming,PyShadowingNames
def getABCD(slh, a0={}, doubled_up=True):
    """
    Return the A, B, C, D and (a, c) matrices that linearize an SLH model
        about a coherent displacement amplitude a0.

    The equations of motion and the input-output relation are then:

    dX = (A X + a) dt + B dA_in
    dA_out = (C X + c) dt + D dA_in

    where, if doubled_up == False
        dX = [a_1, ..., a_m] if doubled_up == False
     and
        dA_in = [dA_1, ..., dA_n]

    or if doubled_up == True
        dX = [a_1, ..., a_m, a_1^*, ... a_m^*]
     and
        dA_in = [dA_1, ..., dA_n, dA_1^*, ..., dA_n^*]

    Params
    ------
    :param slh: SLH object
    :param a0: dictionary of coherent amplitudes {a1: a1_0, a2: a2_0, ...} with annihilation mode operators
        as keys and (numeric or symbolic) amplitude as values.
    :param doubled_up: boolean, necessary for phase-sensitive / active systems

    Returns SymPy matrix objects
    ----------------------------
    A tuple (A, B, C, D, a, c])

    A: coupling of modes to each other
    B: coupling of external input fields to modes
    C: coupling of internal modes to output
    D: coupling of external input fields to output fields

    a: constant coherent input vector for mode e.o.m.
    c: constant coherent input vector of scattered amplitudes contributing to the output
    """

    # the different degrees of freedom
    modes = sorted(slh.space.local_factors())

    # various dimensions
    ncav = len(modes)
    cdim = slh.cdim

    # initialize the matrices
    if doubled_up:
        A = np_zeros((2*ncav, 2*ncav), dtype=object)
        B = np_zeros((2*ncav, 2*cdim), dtype=object)
        C = np_zeros((2*cdim, 2*ncav), dtype=object)
        a = np_zeros(2*ncav, dtype=object)
        c = np_zeros(2*cdim, dtype=object)

    else:
        A = np_zeros((ncav, ncav), dtype=object)
        B = np_zeros((ncav, cdim), dtype=object)
        C = np_zeros((cdim, ncav), dtype=object)
        a = np_zeros(ncav, dtype=object)
        c = np_zeros(cdim, dtype=object)

    def _as_complex(o):
        try:
            return complex(o.coeff) if (isinstance(o, ScalarTimesOperator) and o.term is IdentityOperator) else o
        except TypeError:
            return o

    D = np_array([[_as_complex(o) for o in Sjj] for Sjj in slh.S.matrix])

    if doubled_up:
        # need to explicitly compute D^* because numpy object-dtype array's conjugate() method
        # doesn't work
        Dc = np_array([[D[ii, jj].conjugate() for jj in range(cdim)] for ii in range(cdim)])
        D = np_vstack((np_hstack((D, np_zeros((cdim, cdim)))),
                       np_hstack((np_zeros((cdim, cdim)), Dc))))

    # create substitutions to displace the model
    mode_substitutions = {aj: aj + aj_0 * IdentityOperator for aj, aj_0 in a0.items()}
    mode_substitutions.update({
        aj.dag(): aj.dag() + aj_0.conjugate() * IdentityOperator for aj, aj_0 in a0.items()
    })
    if len(mode_substitutions):
        slh_displaced = slh.substitute(mode_substitutions).expand().simplify_scalar()
    else:
        slh_displaced = slh

    # make symbols for the external field modes
    noises = [OperatorSymbol('b_{{{}}}'.format(n), "ext({})".format(n)) for n in range(cdim)]

    print("computing QSDEs")
    # compute the QSDEs for the internal operators
    eoms = [slh_displaced.symbolic_heisenberg_eom(Destroy(s), noises=noises).expand().simplify_scalar() for s in modes]

    print("Extracting matrices")
    # use the coefficients to generate A, B matrices
    for jj, sjj in enumerate(modes):
        coeffsjj = get_coeffs(eoms[jj])
        a[jj] = coeffsjj[IdentityOperator]
        if doubled_up:
            a[jj+ncav] = coeffsjj[IdentityOperator].conjugate()

        for kk, skk in enumerate(modes):
            A[jj, kk] = coeffsjj[Destroy(skk)]
            if doubled_up:
                A[jj+ncav, kk+ncav] = coeffsjj[Destroy(skk)].conjugate()
                A[jj, kk + ncav] = coeffsjj[Create(skk)]
                A[jj+ncav, kk] = coeffsjj[Create(skk)].conjugate()

        for kk, dAkk in enumerate(noises):
            B[jj, kk] = coeffsjj[dAkk]
            if doubled_up:
                B[jj+ncav, kk+cdim] = coeffsjj[dAkk].conjugate()
                B[jj, kk+cdim] = coeffsjj[dAkk.dag()]
                B[jj + ncav, kk] = coeffsjj[dAkk.dag()].conjugate()

    # use the coefficients in the L vector to generate the C, D
    # matrices
    for jj, Ljj in enumerate(slh_displaced.L.matrix[:, 0]):
        coeffsjj = get_coeffs(Ljj)
        c[jj] = coeffsjj[IdentityOperator]
        if doubled_up:
            c[jj+cdim] + coeffsjj[IdentityOperator].conjugate()

        for kk, skk in enumerate(modes):
            C[jj, kk] = coeffsjj[Destroy(skk)]
            if doubled_up:
                C[jj+cdim, kk+ncav] = coeffsjj[Destroy(skk)].conjugate()
                C[jj, kk+ncav] = coeffsjj[Create(skk)]
                C[jj+cdim, kk] = coeffsjj[Create(skk)].conjugate()

    return map(SympyMatrix, (A, B, C, D, a, c))


# noinspection PyPep8Naming
def move_drive_to_H(slh, which=[]):
    """
    Take an SLH model and replace it by one with the same master equation,
    but where the coherent drive terms have been shifted to the Hamilton operator.


    :param slh: The SLH model to convert.
    :param which: Specifies for which indices this should happen. If [], for all.
    :rtype : SLH
    """
    slh = slh.expand()
    S, L, H = slh.S, slh.L, slh.H
    scalarcs = []
    for jj, Lj in enumerate(L.matrix[:, 0]):
        if not which or jj in which:
            scalarcs.append(-get_coeffs(Lj)[IdentityOperator])
        else:
            scalarcs.append(0)

    return (SLH(identity_matrix(slh.cdim), scalarcs, 0)
            << slh).expand().simplify_scalar()


# noinspection PyPep8Naming
def prepare_adiabatic_limit(slh, k=None):
    """
    Prepare the adiabatic elimination procedure for
        an SLH object with scaling parameter k->\infty

    :param slh: The SLH object to take the limit for
    :param k: The scaling parameter.
    :return:  The objects Y, A, B, F, G, N
        necessary to compute the limiting system.
    :rtype: tuple
    """
    if k is None:
        k = symbols('k', positive=True)
    Ld = slh.L.dag()
    LdL = (Ld * slh.L)[0, 0]
    K = (-LdL / 2 + I * slh.H).expand().simplify_scalar()
    N = slh.S.dag()
    B, A, Y = K.series_expand(k, 0, 2)
    G, F = Ld.series_expand(k, 0, 1)

    return Y, A, B, F, G, N


# noinspection PyPep8Naming
def eval_adiabatic_limit(YABFGN, Ytilde, P0):
    """
    Compute the limiting SLH model for the adiabatic approximation.

    :param YABFGN: The tuple (Y, A, B, F, G, N)
        as returned by prepare_adiabatic_limit.
    :param Ytilde: The pseudo-inverse of Y, satisfying Y * Ytilde = P0.
    :param P0: The projector onto the null-space of Y.
    :return: Limiting SLH model
    :rtype: SLH
    """
    Y, A, B, F, G, N = YABFGN

    Klim = (P0 * (B - A * Ytilde * A) * P0).expand().simplify_scalar()
    Hlim = ((Klim - Klim.dag())/2/I).expand().simplify_scalar()

    Ldlim = (P0 * (G - A * Ytilde * F) * P0).expand().simplify_scalar()

    dN = identity_matrix(N.shape[0]) + F.H * Ytilde * F
    Nlim = (P0 * N * dN * P0).expand().simplify_scalar()

    return SLH(Nlim.dag(), Ldlim.dag(), Hlim.dag())


def _cumsum(lst):
    if not len(lst):
        return []
    sm = lst[0]
    ret = [sm]
    for s in lst[1:]:
        sm += s
        ret += [sm]
    return ret


def connect(components, connections):
    """
    Connect a list of components according to a list of connections.

    :param components: sequence of circuit components
    :param connections: sequence of nested tuples
        ((c1_index, p1_index), (c2_index, p2_index)),
        ((c3_index, p3_index), (c4_index, p4_index)), ...
        specifying connections from component c1, port p1
            to component c2, port c2, etc.
    """
    combined = Concatenation.create(*components)
    cdims = [c.cdim for c in components]
    offsets = _cumsum([0] + cdims[:-1])
    imap = []
    omap = []
    for (c1, op), (c2, ip) in connections:
        op_idx = offsets[c1] + op
        ip_idx = offsets[c2] + ip
        imap.append(ip_idx)
        omap.append(op_idx)
    n = combined.cdim
    nfb = len(connections)
    
    imapping = map_signals_circuit({k: im for k, im
                                    in zip(range(n-nfb, n), imap)}, n)

    omapping = map_signals_circuit({om: k for k, om
                                    in zip(range(n-nfb, n), omap)}, n)

    combined = omapping << combined << imapping
    
    for k in range(nfb):
        combined = combined.feedback()
    return combined





