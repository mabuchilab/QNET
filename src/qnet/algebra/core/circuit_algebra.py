"""Implementation of the SLH circuit algebra

For more details see :ref:`circuit_algebra`.
"""
import os
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import reduce

import numpy as np
import sympy
from sympy import I
from sympy import Matrix as SympyMatrix
from sympy import symbols, sympify

from .abstract_algebra import (
    Expression, Operation, substitute, )
from .algebraic_properties import (
    assoc, check_cdims, filter_neutral, filter_cid, match_replace,
    match_replace_binary)
from .exceptions import (
    AlgebraError, BasisNotSetError, CannotConvertToSLH,
    CannotEliminateAutomatically, CannotVisualize, IncompatibleBlockStructures,
    WrongCDimError)
from .hilbert_space_algebra import LocalSpace, ProductSpace
from .matrix_algebra import (
    Matrix, block_matrix, identity_matrix, permutation_matrix,
    vstackm, zerosm)
from .operator_algebra import (
    IdentityOperator, LocalProjector, LocalSigma, Operator,
    OperatorPlus, OperatorSymbol, ScalarTimesOperator, ZeroOperator, adjoint,
    get_coeffs)
from ...utils.permutations import (
    BadPermutationError, block_perm_and_perms_within_blocks, check_permutation,
    full_block_perm, invert_permutation, permutation_to_block_permutations, )
from ...utils.singleton import Singleton, singleton_object

__all__ = [
    'CPermutation', 'CircuitSymbol', 'Concatenation', 'Feedback',
    'Circuit', 'SLH', 'SeriesInverse', 'SeriesProduct', 'FB',
    'circuit_identity', 'eval_adiabatic_limit',
    'extract_channel', 'getABCD', 'map_channels', 'move_drive_to_H',
    'pad_with_identity', 'prepare_adiabatic_limit',
    'try_adiabatic_elimination', 'CIdentity', 'CircuitZero', 'Component', ]

__private__ = []  # anything not in __all__ must be in __private__


###############################################################################
# Abstract base classes
###############################################################################


class Circuit(metaclass=ABCMeta):
    """Base class for the circuit algebra elements"""

    @property
    @abstractmethod
    def cdim(self) -> int:
        """The channel dimension of the circuit expression,
        i.e. the number of external bosonic noises/inputs that the circuit
        couples to.
        """
        raise NotImplementedError(self.__class__.__name__)

    @property
    def block_structure(self) -> tuple:
        """If the circuit is *reducible* (i.e., it can be represented as a
        :class:`Concatenation` of individual circuit expressions),
        this gives a tuple of cdim values of the subblocks.
        E.g. if A and B are irreducible and have ``A.cdim = 2``, ``B.cdim = 3``

            >>> A = CircuitSymbol('A', cdim=2)
            >>> B = CircuitSymbol('B', cdim=3)

        Then the block structure of their Concatenation is:

            >>> (A + B).block_structure
            (2, 3)

        while

            >>> A.block_structure
            (2,)
            >>> B.block_structure
            (3,)

        See also:

            :meth:`get_blocks` allows to actually retrieve the blocks::

                >>> (A + B).get_blocks()
                (A, B)
        """
        return self._block_structure

    @property
    def _block_structure(self) -> tuple:
        return tuple((self.cdim, ))

    def index_in_block(self, channel_index: int) -> int:
        """Return the index a channel has within the subblock it belongs to

        I.e., only for reducible circuits, this gives a result different from
        the argument itself.

        Args:
            channel_index (int): The index of the external channel

        Raises:
            ValueError: for an invalid `channel_index`

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
        """For a reducible circuit, get a sequence of subblocks that when
        concatenated again yield the original circuit.  The block structure
        given has to be compatible with the circuits actual block structure,
        i.e. it can only be more coarse-grained.

        Args:
            block_structure (tuple): The block structure according to which the
                subblocks are generated (default = ``None``, corresponds to the
                circuit's own block structure)

        Returns:
            A tuple of subblocks that the circuit consists of.

        Raises:
            .IncompatibleBlockStructures
        """
        if block_structure is None:
            block_structure = self.block_structure
        try:
            return self._get_blocks(block_structure)
        except IncompatibleBlockStructures as e:
            raise e

    def _get_blocks(self, block_structure):
        if block_structure == self.block_structure:
            return (self, )
        raise IncompatibleBlockStructures("Requested incompatible block "
                                          "structure %s" % (block_structure,))

    def series_inverse(self) -> 'Circuit':
        """Return the inverse object (under the series product) for a circuit

        In general for any X

            >>> X = CircuitSymbol('X', cdim=3)
            >>> (X << X.series_inverse() == X.series_inverse() << X ==
            ...  circuit_identity(X.cdim))
            True
        """
        return self._series_inverse()

    def _series_inverse(self) -> 'Circuit':
        return SeriesInverse.create(self)

    def feedback(self, *, out_port=None, in_port=None):
        """Return a circuit with self-feedback from the output port
        (zero-based) ``out_port`` to the input port ``in_port``.

        Args:
            out_port (int or None): The output port from which the feedback
                connection leaves (zero-based, default ``None`` corresponds
                to the *last* port).
            in_port (int or None): The input port into which the feedback
                connection goes (zero-based, default ``None`` corresponds to
                the *last* port).
        """
        if out_port is None:
            out_port = self.cdim - 1
        if in_port is None:
            in_port = self.cdim - 1
        return self._feedback(out_port=out_port, in_port=in_port)

    def _feedback(self, *, out_port: int, in_port: int) -> 'Circuit':
        return Feedback.create(self, out_port=out_port, in_port=in_port)

    def show(self):
        """Show the circuit expression in an IPython notebook."""

        # noinspection PyPackageRequirements
        from IPython.display import Image, display

        fname = self.render()
        display(Image(filename=fname))

    def render(self, fname=''):
        """Render the circuit expression and store the result in a file

        Args:
            fname (str): Path to an image file to store the result in.

        Returns:
            str: The path to the image file
        """
        import qnet.visualization.circuit_pyx as circuit_visualization
        from tempfile import gettempdir
        from time import time, sleep

        if not fname:

            tmp_dir = gettempdir()
            fname = os.path.join(tmp_dir, "tmp_{}.png".format(hash(time)))

        if circuit_visualization.draw_circuit(self, fname):
            done = False
            for k in range(20):
                if os.path.exists(fname):

                    done = True
                    break
                else:
                    sleep(.5)
            if done:
                return fname

        raise CannotVisualize()

    def creduce(self) -> 'Circuit':
        """If the circuit is reducible, try to reduce each subcomponent once

        Depending on whether the components at the next hierarchy-level are
        themselves reducible, successive ``circuit.creduce()`` operations
        yields an increasingly fine-grained decomposition of a circuit into its
        most primitive elements.
        """
        return self._creduce()

    @abstractmethod
    def _creduce(self) -> 'Circuit':
        return self

    def toSLH(self) -> 'SLH':
        """Return the SLH representation of a circuit. This can fail if there
        are un-substituted pure circuit symbols (:py:class:`CircuitSymbol`)
        left in the expression"""
        return self._toSLH()

    @abstractmethod
    def _toSLH(self) -> 'SLH':
        raise NotImplementedError(self.__class__.__name__)

    def coherent_input(self, *input_amps) -> 'Circuit':
        """Feed coherent input amplitudes into the circuit.  E.g. For a circuit
        with channel dimension of two, `C.coherent_input(0,1)` leads to an
        input amplitude of zero into the first and one into the second port.

        Args:
            input_amps (SCALAR_TYPES): The coherent input amplitude for each
                port

        Returns:
            Circuit: The circuit including the coherent inputs.

        Raises:
            .WrongCDimError
        """
        return self._coherent_input(*input_amps)

    def _coherent_input(self, *input_amps) -> 'Circuit':
        n_inputs = len(input_amps)
        if n_inputs != self.cdim:
            raise WrongCDimError()
        from qnet.algebra.library.circuit_components import (
                CoherentDriveCC as Displace_cc)

        if n_inputs == 1:
            concat_displacements = Displace_cc(displacement=input_amps[0])
        else:
            displacements = [
                Displace_cc(displacement=amp) if amp != 0
                else circuit_identity(1)
                for amp in input_amps]
            concat_displacements = Concatenation(*displacements)
        return self << concat_displacements

    def __lshift__(self, other):
        if isinstance(other, Circuit):
            return SeriesProduct.create(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Circuit):
            return Concatenation.create(self, other)
        return NotImplemented


###############################################################################
# SLH
###############################################################################


class SLH(Circuit, Expression):
    """Element of the SLH algebra

    The SLH class encapsulate an open system model that is parametrized the a
    scattering matrix (S), a column vector of Lindblad operators (L), and a
    Hamiltonian (H).

    Args:
        S (Matrix): The scattering matrix (with in general Operator-valued
            elements)
        L (Matrix): The coupling vector (with in general Operator-valued
            elements)
        H (Operator): The internal Hamiltonian operator
    """

    def __init__(self, S, L, H):
        if not isinstance(S, Matrix):
            S = Matrix(S)
        if not isinstance(L, Matrix):
            L = Matrix(L)
        if S.shape[0] != L.shape[0]:
            raise ValueError('S and L misaligned: S = {!r}, L = {!r}'
                             .format(S, L))
        if L.shape[1] != 1:
            raise ValueError(("L has wrong shape %s. L must be a column vector "
                              "of operators (shape n × 1)") % str(L.shape))

        if not all(isinstance(s, Operator) for s in S.matrix.ravel()):
            S = S * IdentityOperator
        if not all(isinstance(l, Operator) for l in L.matrix.ravel()):
            L = L * IdentityOperator
        if not isinstance(H, Operator):
            H = H * IdentityOperator

        self.S = S  #: Scattering matrix
        self.L = L  #: Coupling vector
        self.H = H  #: Hamiltonian
        super().__init__(S, L, H)

    @property
    def args(self):
        return self.S, self.L, self.H

    @property
    def Ls(self):
        """Lindblad operators (entries of the L vector), as a list"""
        return list(self.L.matrix[:, 0])

    @property
    def cdim(self):
        """The circuit dimension"""
        return self.S.shape[0]

    def _creduce(self):
        return self

    @property
    def space(self):
        """Total Hilbert space"""
        args_spaces = (self.S.space, self.L.space, self.H.space)
        return ProductSpace.create(*args_spaces)

    @property
    def free_symbols(self):
        """Set of all symbols occcuring in S, L, or H"""
        return set.union(
            self.S.free_symbols, self.L.free_symbols, self.H.free_symbols)

    def series_with_slh(self, other):
        """Series product with another :class:`SLH` object

        Args:
            other (SLH): An upstream SLH circuit.

        Returns:
            SLH: The combined system.
        """
        new_S = self.S * other.S
        new_L = self.S * other.L + self.L

        def ImAdjoint(m):
            return (m.H - m) * (I / 2)

        delta = ImAdjoint(self.L.adjoint() * self.S * other.L)

        if isinstance(delta, Matrix):
            new_H = self.H + other.H + delta[0, 0]
        else:
            assert delta == 0
            new_H = self.H + other.H

        return SLH(new_S, new_L, new_H)

    def concatenate_slh(self, other):
        """Concatenation with another :class:`SLH` object"""
        selfS = self.S
        otherS = other.S
        new_S = block_matrix(
                selfS, zerosm((selfS.shape[0], otherS.shape[1]), dtype=int),
                zerosm((otherS.shape[0], selfS.shape[1]), dtype=int), otherS)
        new_L = vstackm((self.L, other.L))
        new_H = self.H + other.H

        return SLH(new_S, new_L, new_H)

    def _toSLH(self):
        return self

    def expand(self):
        """Expand out all operator expressions within S, L and H

        Return a new :class:`SLH` object with these expanded expressions.
        """
        return SLH(self.S.expand(), self.L.expand(), self.H.expand())

    def simplify_scalar(self, func=sympy.simplify):
        """Simplify all scalar expressions within S, L and H

        Return a new :class:`SLH` object with the simplified expressions.

        See also: :meth:`.QuantumExpression.simplify_scalar`
        """
        return SLH(
            self.S.simplify_scalar(func=func),
            self.L.simplify_scalar(func=func),
            self.H.simplify_scalar(func=func))

    def _series_inverse(self):
        return SLH(self.S.adjoint(), - self.S.adjoint() * self.L, -self.H)

    def _feedback(self, *, out_port, in_port):

        if not isinstance(self.S, Matrix) or not isinstance(self.L, Matrix):
            return Feedback(self, out_port=out_port, in_port=in_port)

        from sympy.core.numbers import ComplexInfinity, Infinity
        sympyOne = sympify(1)

        n = self.cdim - 1

        if out_port != n:
            return (
                map_channels({out_port: n}, self.cdim).toSLH() <<
                self
            ).feedback(in_port=in_port)
        elif in_port != n:
            return (
                self << map_channels({n: in_port}, self.cdim).toSLH()
            ).feedback()

        S, L, H = self.S, self.L, self.H

        one_minus_Snn = sympyOne - S[n, n]

        if isinstance(one_minus_Snn, Operator):
            if one_minus_Snn is IdentityOperator:
                one_minus_Snn = 1
            elif (isinstance(one_minus_Snn, ScalarTimesOperator) and
                  one_minus_Snn.term is IdentityOperator):
                one_minus_Snn = one_minus_Snn.coeff
            else:
                raise AlgebraError('Inversion not implemented for general'
                                   ' operators: {}'.format(one_minus_Snn))

        one_minus_Snn_inv = sympyOne / one_minus_Snn
        if one_minus_Snn_inv in [Infinity, ComplexInfinity]:
            raise AlgebraError(
                "Ill-posed network: singularity in feedback [%s]%d->%d"
                % (str(self), out_port, in_port))

        new_S = S[:n, :n] + S[:n, n:] * one_minus_Snn_inv * S[n:, :n]
        new_L = L[:n] + S[:n, n] * one_minus_Snn_inv * L[n]

        def ImAdjoint(m):
            return (m.H - m) * (I / 2)

        delta_H = ImAdjoint(
            (L.adjoint() * S[:, n:]) * one_minus_Snn_inv * L[n, 0])

        if isinstance(delta_H, Matrix):
            delta_H = delta_H[0, 0]
        new_H = H + delta_H

        return SLH(new_S, new_L, new_H)

    def symbolic_liouvillian(self):
        from qnet.algebra.core.super_operator_algebra import liouvillian

        return liouvillian(self.H, self.L)

    def symbolic_master_equation(self, rho=None):
        """Compute the symbolic Liouvillian acting on a state rho

        If no rho is given, an OperatorSymbol is created in its place.
        This correspnds to the RHS of the master equation
        in which an average is taken over the external noise degrees of
        freedom.

        Args:
            rho (Operator): A symbolic density matrix operator

        Returns:
            Operator: The RHS of the master equation.

        """
        L, H = self.L, self.H
        if rho is None:
            rho = OperatorSymbol('rho', hs=self.space)
        return (-I * (H * rho - rho * H) +
                sum(Lk * rho * adjoint(Lk) -
                    (adjoint(Lk) * Lk * rho + rho * adjoint(Lk) * Lk) / 2
                    for Lk in L.matrix.ravel()))

    def symbolic_heisenberg_eom(
            self, X=None, noises=None, expand_simplify=True):
        """Compute the symbolic Heisenberg equations of motion of a system
        operator X.  If no X is given, an OperatorSymbol is created in its
        place.  If no noises are given, this correspnds to the
        ensemble-averaged Heisenberg equation of motion.

        Args:
            X (Operator): A system operator
            noises (Operator): A vector of noise inputs

        Returns:
            Operator: The RHS of the Heisenberg equations of motion of X.
        """
        L, H = self.L, self.H

        if X is None:
            X = OperatorSymbol('X', hs=(L.space | H.space))

        summands = [I * (H * X - X * H), ]
        for Lk in L.matrix.ravel():
            summands.append(adjoint(Lk) * X * Lk)
            summands.append(-(adjoint(Lk) * Lk * X + X * adjoint(Lk) * Lk) / 2)

        if noises is not None:
            if not isinstance(noises, Matrix):
                noises = Matrix(noises)
            LambdaT = (noises.adjoint().transpose() * noises.transpose()).transpose()
            assert noises.shape == L.shape
            S = self.S
            summands.append((adjoint(noises) * S.adjoint() * (X * L - L * X))
                            .expand()[0, 0])
            summand = (((L.adjoint() * X - X * L.adjoint()) * S * noises)
                       .expand()[0, 0])
            summands.append(summand)
            if len(S.space & X.space):
                comm = (S.adjoint() * X * S - X)
                summands.append((comm * LambdaT).expand().trace())

        ret = OperatorPlus.create(*summands)
        if expand_simplify:
            ret = ret.expand().simplify_scalar()
        return ret

    def __iter__(self):
        return iter((self.S, self.L, self.H))

    def _coherent_input(self, *input_amps):
        return super(SLH, self)._coherent_input(*input_amps).toSLH()


###############################################################################
# Circuit algebra elements
###############################################################################


class CircuitSymbol(Circuit, Expression):
    """Symbolic circuit element

    Args:
        label (str): Label for the symbol
        sym_args (Scalar): optional scalar arguments. With zero `sym_args`, the
            resulting symbol is a constant. With one or more `sym_args`, it
            becomes a function.
        cdim (int): The circuit dimension, that is, the number of I/O lines
    """
    _rx_label = re.compile('^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9().+-]+)?$')

    def __init__(self, label, *sym_args, cdim):
        label = str(label)
        cdim = int(cdim)
        self._label = label
        self._cdim = cdim
        self._sym_args = tuple(sym_args)
        if not self._rx_label.match(label):
            raise ValueError("label '%s' does not match pattern '%s'"
                             % (self.label, self._rx_label.pattern))
        super().__init__(label, *sym_args, cdim=cdim)

    @property
    def label(self):
        return self._label

    @property
    def args(self):
        return (self.label, ) + self._sym_args

    @property
    def kwargs(self):
        return {'cdim': self.cdim}

    @property
    def sym_args(self):
        """Tuple of arguments of the symbol"""
        return self._sym_args

    @property
    def cdim(self):
        """Dimension of circuit"""
        return self._cdim

    def _toSLH(self):
        raise CannotConvertToSLH()

    def _creduce(self):
        return self


class Component(CircuitSymbol, metaclass=ABCMeta):
    """Base class for circuit components

    A circuit component is a :class:`CircuitSymbol` that knows its own SLH
    representation. Consequently, it has a fixed number of I/O
    channels (:attr:`CDIM` class attribute), and a fixed number of named
    arguments. Components only accept keyword arguments.

    Any subclass of :class:`Component` must define all of the class attributes
    listed below, and the :meth:`_toSLH` method that return the :class:`SLH`
    object for the component. Subclasses must also use the
    :func:`.properties_for_args` class decorator::

        @partial(properties_for_args, arg_names='ARGNAMES')

    Args:
        label (str): label for the component. Defaults to :attr:`IDENTIFIER`
        kwargs: values for the parameters in :attr:`ARGNAMES`

    Class Attributes:
        CDIM: the circuit dimension (number of I/O channels)
        PORTSIN: list of names for the input ports of the component
        PORTSOUT: list of names for the output ports of the component
        ARGNAMES: the name of the keyword-arguments for the components
            (excluding ``'label'``)
        DEFAULTS: mapping of keyword-argument names to default values
        IDENTIFIER: the default `label`

    Note:
        The port names defined in :attr:`PORTSIN` and :attr:`PORTSOUT` may be
        used when defining connection via :func:`.connect`.

    See also:
        :mod:`qnet.algebra.library.circuit_components` for example
        :class:`Component` subclasses.
    """

    CDIM = 0
    PORTSIN = ()
    PORTSOUT = ()
    ARGNAMES = ()
    DEFAULTS = {}
    IDENTIFIER = ''

    def __init__(self, *, label=None, **kwargs):
        # since Components are commonly user-defined, we do a quick
        # sanity-check that they've defined the necessary class attributes
        # correctly:
        assert isinstance(self.CDIM, int) and self.CDIM > 0
        assert len(self.PORTSIN) == self.CDIM
        assert all([isinstance(name, str) for name in self.PORTSIN])
        assert len(self.PORTSOUT) == self.CDIM
        assert all([isinstance(name, str) for name in self.PORTSOUT])
        assert len(self.DEFAULTS.keys()) == len(self.ARGNAMES)
        assert all([name in self.DEFAULTS.keys() for name in self.ARGNAMES])
        assert isinstance(self.IDENTIFIER, str) and len(self.IDENTIFIER) > 0
        assert self._has_properties_for_args, \
            "must use properties_for_args class decorater"

        if label is None:
            label = self.IDENTIFIER
        else:
            label = str(label)

        for arg_name in kwargs:
            if arg_name not in self.ARGNAMES:
                raise TypeError(
                    "%s got an unexpected keyword argument '%s'"
                    % (self.__class__.__name__, arg_name))

        self._kwargs = OrderedDict([('label', label)])
        self._minimal_kwargs = OrderedDict()
        if label != self.IDENTIFIER:
            self._minimal_kwargs['label'] = label

        args = []
        for arg_name in self.ARGNAMES:
            val = kwargs.get(arg_name, self.DEFAULTS[arg_name])
            args.append(val)
            self.__dict__['_%s' % arg_name] = val
            # properties_for_args class decorator creates properties for each
            # arg_name
            self._kwargs[arg_name] = val
            if val != self.DEFAULTS[arg_name]:
                self._minimal_kwargs[arg_name] = val

        super().__init__(label, *args, cdim=self.CDIM)

    @property
    def args(self):
        """Empty tuple (no arguments)

        See also:
            :attr:`~CircuitSymbol.sym_args` is a tuple of the keyword argument
            values.
        """
        return ()

    @property
    def kwargs(self):
        """An :class:`.OrderedDict` with the value for the `label` argument, as
        well as any name in :attr:`ARGNAMES`
        """
        return self._kwargs

    @property
    def minimal_kwargs(self):
        """An :class:`.OrderedDict` with the keyword arguments necessary to
        instantiate the component.
        """
        return self._minimal_kwargs

    @abstractmethod
    def _toSLH(self):
        pass


class CPermutation(Circuit, Expression):
    r"""Channel permuting circuit

    This circuit expression is only a rearrangement of input and output fields.
    A channel permutation is given as a tuple of image points. A permutation
    :math:`\sigma \in \Sigma_n` of :math:`n` elements  is often
    represented in the following form

    .. math::
         \begin{pmatrix}
                1           &       2   & \dots &   n       \\
                \sigma(1)   & \sigma(2) & \dots & \sigma(n)
        \end{pmatrix},

    but obviously it is fully sufficient to specify the tuple of images
    :math:`(\sigma(1), \sigma(2), \dots, \sigma(n))`.
    We thus parametrize our permutation circuits only in terms of the image
    tuple. Moreover, we will be working with *zero-based indices*!

    A channel permutation circuit for a given permutation (represented as a
    python tuple of image indices) scatters the :math:`j`-th input field to the
    :math:`\sigma(j)`-th output field.
    """
    simplifications = []
    _block_perms = None

    def __init__(self, permutation):
        self._permutation = permutation
        self._cdim = len(permutation)
        super().__init__(permutation)

    @classmethod
    def create(cls, permutation):
        permutation = tuple(permutation)
        if not check_permutation(permutation):
            raise BadPermutationError(str(permutation))
        if list(permutation) == list(range(len(permutation))):
            return circuit_identity(len(permutation))
        return super().create(permutation)

    @property
    def args(self):
        return (self.permutation, )

    @property
    def block_perms(self):
        """If the circuit is reducible into permutations within subranges of
        the full range of channels, this yields a tuple with the internal
        permutations for each such block.

        :type: tuple
        """
        if not self._block_perms:
            self._block_perms = permutation_to_block_permutations(
                self.permutation)
        return self._block_perms

    @property
    def permutation(self):
        """The permutation image tuple."""
        return self._permutation

    def _toSLH(self):
        return SLH(permutation_matrix(self.permutation),
                   zerosm((self.cdim, 1)), 0)

    @property
    def cdim(self):
        return self._cdim

    def _creduce(self):
        return self

    def series_with_permutation(self, other):
        """Compute the series product with another channel permutation circuit

        Args:
            other (CPermutation):

        Returns:
            Circuit: The composite permutation circuit (could also be the
                identity circuit for n channels)
        """
        combined_permutation = tuple([self.permutation[p]
                                      for p in other.permutation])
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
                raise Exception

            block_perms.append(tuple(current_perm))
            current_perm = []
        return tuple(map(CPermutation.create, block_perms))

    def _factorize_for_rhs(self, rhs):
        """Factorize a channel permutation circuit according the block
        structure of the upstream circuit.  This allows to move as much of the
        permutation as possible *around* a reducible circuit upstream.  It
        basically decomposes

            ``permutation << rhs --> permutation' << rhs' << residual'``

        where rhs' is just a block permutated version of rhs and residual'
        is the maximal part of the permutation that one may move around rhs.

        Args:
            rhs (Circuit): An upstream circuit object

        Returns:
            tuple: new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit

        Raises:
            .BadPermutationError
        """
        block_structure = rhs.block_structure

        block_perm, perms_within_blocks \
            = block_perm_and_perms_within_blocks(self.permutation,
                                                 block_structure)
        fblockp = full_block_perm(block_perm, block_structure)

        if not sorted(fblockp) == list(range(self.cdim)):
            raise BadPermutationError()

        new_rhs_circuit = CPermutation.create(fblockp)
        within_blocks = [CPermutation.create(within_block)
                         for within_block in perms_within_blocks]
        within_perm_circuit = Concatenation.create(*within_blocks)
        rhs_blocks = rhs.get_blocks(block_structure)

        summands = [SeriesProduct.create(within_blocks[p], rhs_blocks[p])
                    for p in invert_permutation(block_perm)]
        permuted_rhs_circuit = Concatenation.create(*summands)

        new_lhs_circuit = (self << within_perm_circuit.series_inverse() <<
                           new_rhs_circuit.series_inverse())

        return new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit

    def _feedback(self, *, out_port, in_port):
        n = self.cdim
        new_perm_circuit = (
            map_channels({out_port: (n - 1)}, n) << self <<
            map_channels({(n - 1): in_port}, n))
        if new_perm_circuit == circuit_identity(n):
            return circuit_identity(n - 1)
        new_perm = list(new_perm_circuit.permutation)
        n_inv = new_perm.index(n - 1)
        new_perm[n_inv] = new_perm[n - 1]

        return CPermutation.create(tuple(new_perm[:-1]))

    def _factor_rhs(self, in_port):
        """With::

            n           := self.cdim
            in_im       := self.permutation[in_port]
            m_{k->l}    := map_signals_circuit({k:l}, n)

        solve the equation (I) containing ``self``::

            self << m_{(n-1) -> in_port}
                == m_{(n-1) -> in_im} << (red_self + cid(1))          (I)

        for the (n-1) channel CPermutation ``red_self``.
        Return in_im, red_self.

        This is useful when ``self`` is the RHS in a SeriesProduct Object that
        is within a Feedback loop as it allows to extract the feedback channel
        from the permutation and moving the remaining part of the permutation
        (``red_self``) outside of the feedback loop.

        :param int in_port: The index for which to factor.
        """
        n = self.cdim
        if not (0 <= in_port < n):
            raise Exception
        in_im = self.permutation[in_port]
        # (I) is equivalent to
        #       m_{in_im -> (n-1)} <<  self << m_{(n-1) -> in_port}
        #           == (red_self + cid(1))     (I')
        red_self_plus_cid1 = (map_channels({in_im: (n - 1)}, n) <<
                              self <<
                              map_channels({(n - 1): in_port}, n))
        if isinstance(red_self_plus_cid1, CPermutation):

            #make sure we can factor
            assert red_self_plus_cid1.permutation[(n - 1)] == (n - 1)

            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])

            return in_im, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_port ->  in_im}

            return in_im, circuit_identity(n - 1)

    def _factor_lhs(self, out_port):
        """With::

            n           := self.cdim
            out_inv     := invert_permutation(self.permutation)[out_port]
            m_{k->l}    := map_signals_circuit({k:l}, n)

        solve the equation (I) containing ``self``::

            m_{out_port -> (n-1)} << self
                == (red_self + cid(1)) << m_{out_inv -> (n-1)}           (I)

        for the (n-1) channel CPermutation ``red_self``.
        Return out_inv, red_self.

        This is useful when 'self' is the LHS in a SeriesProduct Object that is
        within a Feedback loop as it allows to extract the feedback channel
        from the permutation and moving the remaining part of the permutation
        (``red_self``) outside of the feedback loop.
        """
        n = self.cdim
        assert (0 <= out_port < n)
        out_inv = self.permutation.index(out_port)

        # (I) is equivalent to
        #       m_{out_port -> (n-1)} <<  self << m_{(n-1)
        #           -> out_inv} == (red_self + cid(1))     (I')

        red_self_plus_cid1 = (map_channels({out_port: (n - 1)}, n) <<
                              self <<
                              map_channels({(n - 1): out_inv}, n))

        if isinstance(red_self_plus_cid1, CPermutation):

            #make sure we can factor
            assert red_self_plus_cid1.permutation[(n - 1)] == (n - 1)

            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])

            return out_inv, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_port ->  in_im}

            return out_inv, circuit_identity(n - 1)


@singleton_object
class CIdentity(Circuit, Expression, metaclass=Singleton):
    """Single pass-through channel; neutral element of :class:`SeriesProduct`
    """

    _cdim = 1

    @property
    def cdim(self):
        """Dimension of circuit"""
        return self._cdim

    @property
    def args(self):
        return tuple()

    def _toSLH(self):
        return SLH(Matrix([[1]]), Matrix([[0]]), 0)

    def _creduce(self):
        return self

    def _series_inverse(self):
        return self


@singleton_object
class CircuitZero(Circuit, Expression, metaclass=Singleton):
    """Zero circuit, the neutral element of :class:`Concatenation`

    No ports, no internal dynamics."""
    _cdim = 0

    @property
    def cdim(self):
        """Dimension of circuit"""
        return self._cdim

    @property
    def args(self):
        return tuple()

    def _toSLH(self):
        return SLH(Matrix([[]]), Matrix([[]]), 0)

    def _creduce(self):
        return self


###############################################################################
# Algebra Operations
###############################################################################


class SeriesProduct(Circuit, Operation):
    """The series product circuit operation. It can be applied to any sequence
    of circuit objects that have equal channel dimension.
    """
    simplifications = [assoc, filter_cid, check_cdims,
                       match_replace_binary]
    _binary_rules = OrderedDict()  # see end of module

    _neutral_element = CIdentity

    neutral_element = _neutral_element

    @property
    def cdim(self):
        return self.operands[0].cdim

    def _toSLH(self):
        return reduce(lambda a, b: a.toSLH().series_with_slh(b.toSLH()),
                      self.operands)

    def _creduce(self):
        return SeriesProduct.create(*[op.creduce() for op in self.operands])

    def _series_inverse(self):
        factors = [o.series_inverse() for o in reversed(self.operands)]
        return SeriesProduct.create(*factors)


class Concatenation(Circuit, Operation):
    """Concatenation of circuit elements"""

    simplifications = [assoc, filter_neutral, match_replace_binary]

    _binary_rules = OrderedDict()  # see end of module

    _neutral_element = CircuitZero

    neutral_element = _neutral_element

    def __init__(self, *operands):
        self._cdim = None
        super().__init__(*operands)

    @property
    def cdim(self):
        """Circuit dimension (sum of dimensions of the operands)"""
        if self._cdim is None:
            self._cdim = sum((circuit.cdim for circuit in self.operands))
        return self._cdim

    def _toSLH(self):
        return reduce(lambda a, b:
                      a.toSLH().concatenate_slh(b.toSLH()), self.operands)

    def _creduce(self):
        return Concatenation.create(*[op.creduce() for op in self.operands])

    @property
    def _block_structure(self):
        result = []
        for op in self.operands:
            op_structure = op.block_structure
            result.extend(op_structure)
        return tuple(result)

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
                raise IncompatibleBlockStructures(
                    'requested blocks according to incompatible '
                    'block_structure')
            blocks.append(Concatenation.create(*cbo))
            cbo = []
            current_length = 0
        return tuple(blocks)

    def _series_inverse(self):
        return Concatenation.create(*[o.series_inverse()
                                      for o in self.operands])

    def _feedback(self, *, out_port, in_port):

        n = self.cdim

        if out_port == n - 1 and in_port == n - 1:
            return Concatenation.create(*(self.operands[:-1] +
                                          (self.operands[-1].feedback(),)))

        in_port_in_block, in_block = self.index_in_block(in_port)
        out_index_in_block, out_block = self.index_in_block(out_port)

        blocks = self.get_blocks()

        if in_block == out_block:
            return (Concatenation.create(*blocks[:out_block]) +
                    blocks[out_block].feedback(
                           out_port=out_index_in_block,
                           in_port=in_port_in_block) +
                    Concatenation.create(*blocks[out_block + 1:]))
        ### no 'real' feedback loop, just an effective series
        #partition all blocks into just two

        if in_block < out_block:
            b1 = Concatenation.create(*blocks[:out_block])
            b2 = Concatenation.create(*blocks[out_block:])

            return ((b1 + circuit_identity(b2.cdim - 1)) <<
                    map_channels({out_port - 1: in_port}, n - 1) <<
                    (circuit_identity(b1.cdim - 1) + b2))
        else:
            b1 = Concatenation.create(*blocks[:in_block])
            b2 = Concatenation.create(*blocks[in_block:])

            return ((circuit_identity(b1.cdim - 1) + b2) <<
                    map_channels({out_port: in_port - 1}, n - 1) <<
                    (b1 + circuit_identity(b2.cdim - 1)))


class Feedback(Circuit, Operation):
    """Feedback on a single channel of a circuit

    The circuit feedback operation applied to a circuit of channel
    dimension > 1 and from an output port index to an input port index.

    Args:
        circuit (Circuit): The circuit that undergoes self-feedback
        out_port (int): The output port index.
        in_port (int): The input port index.
    """
    delegate_to_method = (Concatenation, SLH, CPermutation)

    simplifications = [match_replace, ]

    _rules = OrderedDict()  # see end of module

    def __init__(self, circuit: Circuit, *, out_port: int, in_port: int):
        self.out_port = int(out_port)
        self.in_port = int(in_port)
        operands = [circuit, ]
        super().__init__(*operands)

    @property
    def kwargs(self):
        return OrderedDict([('out_port', self.out_port),
                            ('in_port', self.in_port)])

    @property
    def operand(self):
        """The Circuit that undergoes feedback"""
        return self._operands[0]

    @property
    def out_in_pair(self):
        """Tuple of zero-based feedback port indices (out_port, in_port)"""
        return (self.out_port, self.in_port)

    @property
    def cdim(self):
        """Circuit dimension (one less than the circuit on which the feedback
        acts"""
        return self.operand.cdim - 1

    @classmethod
    def create(cls, circuit: Circuit, *, out_port: int, in_port: int) \
            -> 'Feedback':
        if not isinstance(circuit, Circuit):
            raise ValueError()

        n = circuit.cdim
        if n <= 1:
            raise ValueError("circuit dimension %d needs to be > 1 in order "
                             "to apply a feedback" % n)

        if isinstance(circuit, cls.delegate_to_method):
            return circuit._feedback(out_port=out_port, in_port=in_port)

        return super().create(circuit, out_port=out_port, in_port=in_port)

    def _toSLH(self):
        return self.operand.toSLH().feedback(
                out_port=self.out_port, in_port=self.in_port)

    def _creduce(self):
        return self.operand.creduce().feedback(
                out_port=self.out_port, in_port=self.in_port)

    def _series_inverse(self):
        return Feedback.create(self.operand.series_inverse(),
                               in_port=self.out_port, out_port=self.in_port)


class SeriesInverse(Circuit, Operation):
    """Symbolic series product inversion operation

        ``SeriesInverse(circuit)``

    One generally has

        >>> C = CircuitSymbol('C', cdim=3)
        >>> SeriesInverse(C) << C == circuit_identity(C.cdim)
        True

    and

        >>> C << SeriesInverse(C) == circuit_identity(C.cdim)
        True
    """
    simplifications = []
    delegate_to_method = (SeriesProduct, Concatenation, Feedback, SLH,
                          CPermutation, CIdentity.__class__)

    @property
    def operand(self):
        """The un-inverted circuit"""
        return self.operands[0]

    @classmethod
    def create(cls, circuit):
        if isinstance(circuit, SeriesInverse):
            return circuit.operand
        elif isinstance(circuit, cls.delegate_to_method):
            return circuit._series_inverse()
        return super().create(circuit)

    @property
    def cdim(self):
        return self.operand.cdim

    def _toSLH(self):
        return self.operand.toSLH().series_inverse()

    def _creduce(self):
        return self.operand.creduce().series_inverse()


###############################################################################
# Constructor Routines
###############################################################################


def circuit_identity(n):
    """Return the circuit identity for n channels

    Args:
        n (int): The channel dimension

    Returns:
        Circuit: n-channel identity circuit
    """
    if n <= 0:
        return CircuitZero
    if n == 1:
        return CIdentity
    return Concatenation(*((CIdentity,) * n))



def FB(circuit, *, out_port=None, in_port=None):
    """Wrapper for :class:`.Feedback`, defaulting to last channel

    Args:
        circuit (Circuit): The circuit that undergoes self-feedback
        out_port (int): The output port index, default = None --> last port
        in_port (int): The input port index, default = None --> last port

    Returns:
        Circuit: The circuit with applied feedback operation.
    """
    if out_port is None:
        out_port = circuit.cdim - 1
    if in_port is None:
        in_port = circuit.cdim - 1
    return Feedback.create(circuit, out_port=out_port, in_port=in_port)


###############################################################################
# Auxilliary routines
###############################################################################


def extract_channel(k, cdim):
    """Create a :class:`CPermutation` that extracts channel `k`

    Return a permutation circuit that maps the k-th (zero-based)
    input to the last output, while preserving the relative order of all other
    channels.

    Args:
        k (int): Extracted channel index
        cdim (int): The circuit dimension (number of channels)

    Returns:
        Circuit: Permutation circuit
    """
    n = cdim
    perm = tuple(list(range(k)) + [n - 1] + list(range(k, n - 1)))
    return CPermutation.create(perm)


def map_channels(mapping, cdim):
    """Create a :class:`CPermuation` based on a dict of channel mappings

    For a given mapping in form of a dictionary, generate the channel
    permutating circuit that achieves the specified mapping while leaving the
    relative order of all non-specified channels intact.

    Args:
        mapping (dict): Input-output mapping of indices (zero-based)
            ``{in1:out1, in2:out2,...}``
        cdim (int): The circuit dimension (number of channels)

    Returns:
        CPermutation: Circuit mapping the channels as specified
    """
    n = cdim
    free_values = list(range(n))

    for v in mapping.values():
        if v >= n:
            raise ValueError('the mapping cannot take on values larger than '
                             'cdim - 1')
        free_values.remove(v)
    for k in mapping:
        if k >= n:
            raise ValueError('the mapping cannot map keys larger than '
                             'cdim - 1')
    permutation = []
    for k in range(n):
        if k in mapping:
            permutation.append(mapping[k])
        else:
            permutation.append(free_values.pop(0))

    return CPermutation.create(tuple(permutation))


def pad_with_identity(circuit, k, n):
    """Pad a circuit by adding a `n`-channel identity circuit at index `k`

    That is, a circuit of channel dimension $N$ is extended to one of channel
    dimension $N+n$, where the channels $k$, $k+1$, ...$k+n-1$, just pass
    through the system unaffected.  E.g. let ``A``, ``B`` be two single channel
    systems::

        >>> A = CircuitSymbol('A', cdim=1)
        >>> B = CircuitSymbol('B', cdim=1)
        >>> print(ascii(pad_with_identity(A+B, 1, 2)))
        A + cid(2) + B

    This method can also be applied to irreducible systems, but in that case
    the result can not be decomposed as nicely.

    Args:
        circuit (Circuit): circuit to pad
        k (int): The index at which to insert the circuit
        n (int): The number of channels to pass through

    Returns:
        Circuit: An extended circuit that passes through the channels
            $k$, $k+1$, ..., $k+n-1$

    """
    circuit_n = circuit.cdim
    combined_circuit = circuit + circuit_identity(n)
    permutation = (list(range(k)) + list(range(circuit_n, circuit_n + n)) +
                   list(range(k, circuit_n)))
    return (CPermutation.create(invert_permutation(permutation)) <<
            combined_circuit << CPermutation.create(permutation))


def getABCD(slh, a0=None, doubled_up=True):
    """Calculate the ABCD-linearization of an SLH model

    Return the A, B, C, D and (a, c) matrices that linearize an SLH model
    about a coherent displacement amplitude a0.

    The equations of motion and the input-output relation are then:

    dX = (A X + a) dt + B dA_in
    dA_out = (C X + c) dt + D dA_in

    where, if doubled_up == False

        dX = [a_1, ..., a_m]
        dA_in = [dA_1, ..., dA_n]

    or if doubled_up == True

        dX = [a_1, ..., a_m, a_1^*, ... a_m^*]
        dA_in = [dA_1, ..., dA_n, dA_1^*, ..., dA_n^*]

    Args:
        slh: SLH object
        a0: dictionary of coherent amplitudes ``{a1: a1_0, a2: a2_0, ...}``
            with annihilation mode operators as keys and (numeric or symbolic)
            amplitude as values.
        doubled_up: boolean, necessary for phase-sensitive / active systems


    Returns:

        A tuple (A, B, C, D, a, c])

        with

        * `A`: coupling of modes to each other
        * `B`: coupling of external input fields to modes
        * `C`: coupling of internal modes to output
        * `D`: coupling of external input fields to output fields

        * `a`: constant coherent input vector for mode e.o.m.
        * `c`: constant coherent input vector of scattered amplitudes
            contributing to the output
    """
    from qnet.algebra.library.fock_operators import Create, Destroy
    if a0 is None:
        a0 = {}

    # the different degrees of freedom
    full_space = ProductSpace.create(slh.S.space, slh.L.space, slh.H.space)
    modes = sorted(full_space.local_factors)

    # various dimensions
    ncav = len(modes)
    cdim = slh.cdim

    # initialize the matrices
    if doubled_up:
        A = np.zeros((2*ncav, 2*ncav), dtype=object)
        B = np.zeros((2*ncav, 2*cdim), dtype=object)
        C = np.zeros((2*cdim, 2*ncav), dtype=object)
        a = np.zeros(2*ncav, dtype=object)
        c = np.zeros(2*cdim, dtype=object)

    else:
        A = np.zeros((ncav, ncav), dtype=object)
        B = np.zeros((ncav, cdim), dtype=object)
        C = np.zeros((cdim, ncav), dtype=object)
        a = np.zeros(ncav, dtype=object)
        c = np.zeros(cdim, dtype=object)

    def _as_complex(o):
        if isinstance(o, Operator):
            o = o.expand()
            if o is IdentityOperator:
                o = 1
            elif o is ZeroOperator:
                o = 0
            elif isinstance(o, ScalarTimesOperator):
                assert o.term is IdentityOperator
                o = o.coeff
            else:
                raise ValueError("{} is not trivial operator".format(o))
        try:
            return complex(o)
        except TypeError:
            return o

    D = np.array([[_as_complex(o) for o in Sjj] for Sjj in slh.S.matrix])

    if doubled_up:
        # need to explicitly compute D^* because numpy object-dtype array's
        # conjugate() method doesn't work
        Dc = np.array([[D[ii, jj].conjugate() for jj in range(cdim)]
                       for ii in range(cdim)])
        D = np.vstack((np.hstack((D, np.zeros((cdim, cdim)))),
                       np.hstack((np.zeros((cdim, cdim)), Dc))))

    # create substitutions to displace the model
    mode_substitutions = {aj: aj + aj_0 * IdentityOperator
                          for aj, aj_0 in a0.items()}
    mode_substitutions.update({
        aj.dag(): aj.dag() + aj_0.conjugate() * IdentityOperator
        for aj, aj_0 in a0.items()
    })
    if len(mode_substitutions):
        slh_displaced = (slh.substitute(mode_substitutions).expand()
                         .simplify_scalar())
    else:
        slh_displaced = slh

    # make symbols for the external field modes
    noises = [OperatorSymbol('b_{}'.format(n), hs="ext_{}".format(n))
              for n in range(cdim)]

    # compute the QSDEs for the internal operators
    eoms = [slh_displaced.symbolic_heisenberg_eom(Destroy(hs=s), noises=noises)
            for s in modes]

    # use the coefficients to generate A, B matrices
    for jj in range(len(modes)):
        coeffsjj = get_coeffs(eoms[jj])
        a[jj] = coeffsjj[IdentityOperator]
        if doubled_up:
            a[jj+ncav] = coeffsjj[IdentityOperator].conjugate()

        for kk, skk in enumerate(modes):
            A[jj, kk] = coeffsjj[Destroy(hs=skk)]
            if doubled_up:
                A[jj+ncav, kk+ncav] = coeffsjj[Destroy(hs=skk)].conjugate()
                A[jj, kk + ncav] = coeffsjj[Create(hs=skk)]
                A[jj+ncav, kk] = coeffsjj[Create(hs=skk)].conjugate()

        for kk, dAkk in enumerate(noises):
            B[jj, kk] = coeffsjj[dAkk]
            if doubled_up:
                B[jj+ncav, kk+cdim] = coeffsjj[dAkk].conjugate()
                B[jj, kk+cdim] = coeffsjj[dAkk.dag()]
                B[jj + ncav, kk] = coeffsjj[dAkk.dag()].conjugate()

    # use the coefficients in the L vector to generate the C, D
    # matrices
    for jj, Ljj in enumerate(slh_displaced.Ls):
        coeffsjj = get_coeffs(Ljj)
        c[jj] = coeffsjj[IdentityOperator]
        if doubled_up:
            c[jj+cdim] = coeffsjj[IdentityOperator].conjugate()

        for kk, skk in enumerate(modes):
            C[jj, kk] = coeffsjj[Destroy(hs=skk)]
            if doubled_up:
                C[jj+cdim, kk+ncav] = coeffsjj[Destroy(hs=skk)].conjugate()
                C[jj, kk+ncav] = coeffsjj[Create(hs=skk)]
                C[jj+cdim, kk] = coeffsjj[Create(hs=skk)].conjugate()

    return map(SympyMatrix, (A, B, C, D, a, c))


def move_drive_to_H(slh, which=None, expand_simplify=True):
    r'''Move coherent drives from the Lindblad operators to the Hamiltonian.

    For the given SLH model, move inhomogeneities in the Lindblad operators (resulting
    from the presence of a coherent drive, see :class:`CoherentDriveCC`) to the
    Hamiltonian.

    This exploits the invariance of the Lindblad master equation under the
    transformation  (cf. Breuer and Pettrucione, Ch 3.2.1)

    .. math::
        :nowrap:

        \begin{align}
            \Op{L}_i &\longrightarrow \Op{L}_i' = \Op{L}_i - \alpha_i  \\
            \Op{H}   &\longrightarrow
            \Op{H}' = \Op{H} + \frac{1}{2i} \sum_j
                    (\alpha_j \Op{L}_j^{\dagger} - \alpha_j^* \Op{L}_j)
        \end{align}

    In the context of SLH, this transformation is achieved by feeding `slh` into

    .. math::
        \SLH(\identity, -\mat{\alpha}, 0)

    where $\mat{\alpha}$ has the elements $\alpha_i$.

    Parameters
    ----------
    slh : SLH
        SLH model to transform. If `slh` does not contain any inhomogeneities, it is
        invariant under the transformation.

    which : sequence or None
        Sequence of circuit dimensions to apply the transform to. If None, all
        dimensions are transformed.

    expand_simplify : bool
        if True, expand and simplify the new SLH object before returning. This has no
        effect if `slh` does not contain any inhomogeneities.

    Returns
    -------
    new_slh : SLH
        Transformed SLH model.

    '''
    if which is None:
        which = []
    scalarcs = []
    for jj, L in enumerate(slh.Ls):
        if not which or jj in which:
            scalarcs.append(-get_coeffs(L.expand())[IdentityOperator])
        else:
            scalarcs.append(0)

    if np.all(np.array(scalarcs) == 0):
        return slh
    new_slh = SLH(identity_matrix(slh.cdim), scalarcs, 0) << slh
    if expand_simplify:
        return new_slh.expand().simplify_scalar()
    return new_slh


def prepare_adiabatic_limit(slh, k=None):
    """Prepare the adiabatic elimination on an SLH object

    Args:
        slh: The SLH object to take the limit for
        k: The scaling parameter $k \rightarrow \infty$. The default is a
            positive symbol 'k'

    Returns:
        tuple: The objects ``Y, A, B, F, G, N``
        necessary to compute the limiting system.
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


def eval_adiabatic_limit(YABFGN, Ytilde, P0):
    """Compute the limiting SLH model for the adiabatic approximation

    Args:
        YABFGN: The tuple (Y, A, B, F, G, N)
            as returned by prepare_adiabatic_limit.
        Ytilde: The pseudo-inverse of Y, satisfying Y * Ytilde = P0.
        P0: The projector onto the null-space of Y.

    Returns:
        SLH: Limiting SLH model
    """
    Y, A, B, F, G, N = YABFGN

    Klim = (P0 * (B - A * Ytilde * A) * P0).expand().simplify_scalar()
    Hlim = ((Klim - Klim.dag())/2/I).expand().simplify_scalar()

    Ldlim = (P0 * (G - A * Ytilde * F) * P0).expand().simplify_scalar()

    dN = identity_matrix(N.shape[0]) + F.H * Ytilde * F
    Nlim = (P0 * N * dN * P0).expand().simplify_scalar()

    return SLH(Nlim.dag(), Ldlim.dag(), Hlim.dag())


def try_adiabatic_elimination(slh, k=None, fock_trunc=6, sub_P0=True):
    """Attempt to automatically do adiabatic elimination on an SLH object

    This will project the `Y` operator onto a truncated basis with dimension
    specified by `fock_trunc`.  `sub_P0` controls whether an attempt is made to
    replace the kernel projector P0 by an :class:`.IdentityOperator`.
    """
    ops = prepare_adiabatic_limit(slh, k)
    Y = ops[0]
    if isinstance(Y.space, LocalSpace):
        try:
            b = Y.space.basis_labels
            if len(b) > fock_trunc:
                b = b[:fock_trunc]
        except BasisNotSetError:
            b = range(fock_trunc)
        projectors = set(LocalProjector(ll, hs=Y.space) for ll in b)
        Id_trunc = sum(projectors, ZeroOperator)
        Yprojection = (
            ((Id_trunc * Y).expand() * Id_trunc)
            .expand().simplify_scalar())
        termcoeffs = get_coeffs(Yprojection)
        terms = set(termcoeffs.keys())

        for term in terms - projectors:
            cannot_eliminate = (
                not isinstance(term, LocalSigma) or
                not term.operands[1] == term.operands[2])
            if cannot_eliminate:
                raise CannotEliminateAutomatically(
                    "Proj. Y operator has off-diagonal term: ~{}".format(term))
        P0 = sum(projectors - terms, ZeroOperator)
        if P0 == ZeroOperator:
            raise CannotEliminateAutomatically("Empty null-space of Y!")

        Yinv = sum(t/termcoeffs[t] for t in terms & projectors)
        assert (
            (Yprojection*Yinv).expand().simplify_scalar() ==
            (Id_trunc - P0).expand())
        slhlim = eval_adiabatic_limit(ops, Yinv, P0)

        if sub_P0:
            # TODO for non-unit rank P0, this will not work
            slhlim = slhlim.substitute(
                {P0: IdentityOperator}).expand().simplify_scalar()
        return slhlim

    else:
        raise CannotEliminateAutomatically(
            "Currently only single degree of freedom Y-operators supported")


def _cumsum(lst):
    if not len(lst):
        return []
    sm = lst[0]
    ret = [sm]
    for s in lst[1:]:
        sm += s
        ret += [sm]
    return ret
