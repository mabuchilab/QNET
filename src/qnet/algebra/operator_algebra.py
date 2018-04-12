r"""
This module features classes and functions to define and manipulate symbolic
Operator expressions.  For more details see :ref:`operator_algebra`.

For a list of all properties and methods of an operator object, see the
documentation for the basic :py:class:`Operator` class.
"""
import re
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import defaultdict, OrderedDict
from itertools import product as cartesian_product

from sympy import (
    exp, sqrt, I, sympify, Basic as SympyBasic, series as sympy_series)

from .scalar_types import SCALAR_TYPES
from .abstract_algebra import (
    Expression, Operation, ScalarTimesExpression, IndexedSum, assoc,
    assoc_indexed, indexed_sum_over_const, indexed_sum_over_kronecker, orderby,
    filter_neutral, match_replace_binary, match_replace, CannotSimplify,
    check_rules_dict, _scalar_free_symbols, all_symbols)
from ..utils.singleton import Singleton, singleton_object
from .hilbert_space_algebra import (
    TrivialSpace, HilbertSpace, LocalSpace, ProductSpace, BasisNotSetError)
from .pattern_matching import wc, pattern_head, pattern
from ..utils.ordering import (
    KeyTuple, DisjunctCommutativeHSOrder, FullCommutativeHSOrder)
from ..utils.indices import SymbolicLabelBase, KroneckerDelta, IndexRangeBase

sympyOne = sympify(1)

# for hilbert space dimensions less than or equal to this,
# compute numerically PseudoInverse and NullSpaceProjector representations
DENSE_DIMENSION_LIMIT = 1000

__all__ = [
    'Adjoint', 'Create', 'Destroy', 'Displace', 'Jminus', 'Jplus',
    'Jz', 'LocalOperator', 'LocalSigma', 'NullSpaceProjector', 'Operator',
    'OperatorOperation', 'OperatorPlus', 'OperatorPlusMinusCC',
    'OperatorSymbol', 'OperatorTimes', 'OperatorTrace',
    'Phase', 'PseudoInverse', 'ScalarTimesOperator', 'SingleOperatorOperation',
    'Squeeze', 'Jmjmcoeff', 'Jpjmcoeff', 'Jzjmcoeff', 'LocalProjector',
    'X', 'Y', 'Z', 'adjoint', 'create_operator_pm_cc', 'decompose_space',
    'expand_operator_pm_cc', 'factor_coeff', 'factor_for_trace', 'get_coeffs',
    'simplify_scalar', 'space', 'II', 'IdentityOperator', 'ZeroOperator',
    'Commutator', 'OperatorIndexedSum', 'tr']

__private__ = [  # anything not in __all__ must be in __private__
    'implied_local_space', 'delegate_to_method', 'disjunct_hs_zero',
    'scalars_to_op', 'commutator_order']


###############################################################################
# Algebraic properties
###############################################################################


def implied_local_space(*, arg_index=None, keys=None):
    """Return a simplification that converts the positional argument
    `arg_index` from (str, int) to LocalSpace, as well as any keyword argument
    with one of the given keys"""

    def args_to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected args to LocalSpace"""
        if isinstance(args[arg_index], LocalSpace):
            new_args = args
        else:
            if isinstance(args[arg_index], (int, str)):
                hs = LocalSpace(args[arg_index])
            else:
                hs = args[arg_index]
                assert isinstance(hs, HilbertSpace)
            new_args = (tuple(args[:arg_index]) + (hs,) +
                        tuple(args[arg_index + 1:]))
        return new_args, kwargs

    def kwargs_to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected kwargs to LocalSpace"""
        if all([isinstance(kwargs[key], LocalSpace) for key in keys]):
            new_kwargs = kwargs
        else:
            new_kwargs = {}
            for key, val in kwargs.items():
                if key in keys:
                    if isinstance(val, (int, str)):
                        val = LocalSpace(val)
                    assert isinstance(val, HilbertSpace)
                new_kwargs[key] = val
        return args, new_kwargs

    def to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected args and kwargs to LocalSpace"""
        new_args, __ = args_to_local_space(args, kwargs, arg_index)
        __, new_kwargs = kwargs_to_local_space(args, kwargs, keys)
        return new_args, new_kwargs

    if (arg_index is not None) and (keys is None):
        return args_to_local_space
    elif (arg_index is None) and (keys is not None):
        return kwargs_to_local_space
    elif (arg_index is not None) and (keys is not None):
        return to_local_space
    else:
        raise ValueError("must give at least one of arg_index and keys")


def delegate_to_method(mtd):
    """Create a simplification rule that delegates the instantiation to the
    method `mtd` of the operand (if defined)"""

    def _delegate_to_method(cls, ops, kwargs):
        assert len(ops) == 1
        op, = ops
        if hasattr(op, mtd):
            return getattr(op, mtd)()
        else:
            return ops, kwargs

    return _delegate_to_method


def scalars_to_op(cls, ops, kwargs):
    r'''Convert any scalar $\alpha$ in `ops` into an operator $\alpha
    \identity$'''
    op_ops = []
    for op in ops:
        if isinstance(op, SCALAR_TYPES):
            op_ops.append(ScalarTimesOperator.create(op, IdentityOperator))
        else:
            op_ops.append(op)
    return op_ops, kwargs


def disjunct_hs_zero(cls, ops, kwargs):
    """Return ZeroOperator if all the operators in `ops` have a disjunct
    Hilbert space, or an unchanged `ops`, `kwargs` otherwise
    """
    hilbert_spaces = []
    for op in ops:
        try:
            hs = op.space
        except AttributeError:  # scalars
            hs = TrivialSpace
        for hs_prev in hilbert_spaces:
            if not hs.isdisjoint(hs_prev):
                return ops, kwargs
        hilbert_spaces.append(hs)
    return ZeroOperator


def commutator_order(cls, ops, kwargs):
    """Apply anti-commutative property of the commutator to apply a standard
    ordering of the commutator arguments
    """
    assert len(ops) == 2
    if cls.order_key(ops[1]) < cls.order_key(ops[0]):
        return -1 * Commutator.create(ops[1], ops[0])
    else:
        return ops, kwargs


###############################################################################
# Abstract base classes
###############################################################################


class Operator(metaclass=ABCMeta):
    """The basic operator class, which fixes the abstract interface of operator
    objects and where possible also defines the default behavior under
    operations.  Any operator contains an associated HilbertSpace object,
    on which it is taken to act non-trivially.
    """

    @abstractproperty
    def space(self):
        """The :class:`HilbertSpace` on which the operator acts
        non-trivially"""
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """The Hermitian adjoint of the operator."""
        return Adjoint.create(self)

    conjugate = dag = adjoint

    def pseudo_inverse(self):
        """The pseudo-Inverse of the Operator, i.e., it inverts the operator on
        the orthogonal complement of its nullspace"""
        return PseudoInverse.create(self)

    def expand(self):
        """Expand out distributively all products of sums. Note that this does
        not expand out sums of scalar coefficients.

        :return: A fully expanded sum of operators.
        :rtype: Operator
        """
        return self._expand()

    def expand_in_basis(self, basis_states=None, hermitian=False):
        """Write the operator as an expansion into all :class:`KetBras
        <qnet.algebra.state_algebra.KetBra>` spanned by `basis_states`.

        Args:
            basis_states (list or None): List of basis states
                (:class:`~qnet.algebra.state_algebra.Ket` instances) into which
                to expand the operator. If None, use the operator's
                `space.basis_states`
            hermitian (bool): If True, assume that the operator is Hermitian
                and represent all elements in the lower triangle of the
                expansion via :class:`OperatorPlusMinusCC`. This is meant to
                enhance readability

        Raises:
            BasisNotSetError: If `basis_states` is None and the operator's
                Hilbert space has no well-defined basis

        Example:

            >>> hs = LocalSpace(1, basis=('g', 'e'))
            >>> op = LocalSigma('g', 'e', hs=hs) + LocalSigma('e', 'g', hs=hs)
            >>> from qnet.printing import ascii
            >>> print(ascii(op, sig_as_ketbra=False))
            sigma_e,g^(1) + sigma_g,e^(1)
            >>> print(ascii(op.expand_in_basis()))
            |e><g|^(1) + |g><e|^(1)
            >>> print(ascii(op.expand_in_basis(hermitian=True)))
            |g><e|^(1) + c.c.
        """
        from qnet.algebra.state_algebra import KetBra  # avoid circ. import
        if basis_states is None:
            basis_states = list(self.space.basis_states)
        else:
            basis_states = list(basis_states)
        diag_terms = []
        terms = []
        for i, ket_i in enumerate(basis_states):
            for j, ket_j in enumerate(basis_states):
                if i > j and hermitian:
                    continue
                op_ij = (ket_i.dag * self * ket_j).expand()
                ketbra = KetBra(ket_i, ket_j)
                term = op_ij * ketbra
                if term is not ZeroOperator:
                    if i == j:
                        diag_terms.append(op_ij * ketbra)
                    else:
                        terms.append(op_ij * ketbra)
        if hermitian:
            res = OperatorPlus(*diag_terms)
            if len(terms) > 0:
                res = res + OperatorPlusMinusCC(OperatorPlus(*terms))
            return res
        else:
            return OperatorPlus(*diag_terms) + OperatorPlus(*terms)

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

    def simplify_scalar(self):
        """Simplify all scalar coefficients within the Operator expression.

        :return: The simplified expression.
        :rtype: Operator
        """
        return self._simplify_scalar()

    def _simplify_scalar(self):
        return self

    def diff(self, sym, n=1, expand_simplify=True):
        """Differentiate by scalar parameter sym.

        :param ~sympy.cor.symbol.Symbol sym: What to differentiate by.
        :param int n: How often to differentiate
        :param bool expand_simplify: Whether to simplify the result.
        :return (Operator): The n-th derivative.
        """
        expr = self
        for k in range(n):
            expr = expr._diff(sym)
        if expand_simplify:
            expr = expr.expand().simplify_scalar()
        return expr

    def _diff(self, sym):
        return ZeroOperator

    def series_expand(self, param, about, order):
        """Expand the operator expression as a truncated power series in a
        scalar parameter.

        Args:
            param (~sympy.core.symbol.Symbol): Expansion parameter
            about (SCALAR_TYPES): Point about which to expand
            order (int): Maximum order of expansion (>= 0)

        Returns:
            tuple of length ``order + 1``, where the entries are the
            expansion coefficients (instances of :class:`Operator`)
        """
        return self._series_expand(param, about, order)

    @abstractmethod
    def _series_expand(self, param, about, order):
        raise NotImplementedError(self.__class__.__name__)

    def __add__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return OperatorPlus.create(self, other * IdentityOperator)
        elif isinstance(other, Operator):
            return OperatorPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return ScalarTimesOperator.create(other, self)
        elif isinstance(other, Operator):
            return OperatorTimes.create(self, other)
        try:
            return super().__mul__(other)
        except AttributeError:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return ScalarTimesOperator.create(other, self)
        try:
            return super().__rmul__(other)
        except AttributeError:
            return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self * (sympyOne / other)
        try:
            return super().__rmul__(other)
        except AttributeError:
            return NotImplemented

    __truediv__ = __div__

    def __pow__(self, other):
        if isinstance(other, int):
            return OperatorTimes.create(*[self for __ in range(other)])
        try:
            return super().__pow__(other)
        except AttributeError:
            return NotImplemented


class LocalOperator(Operator, Expression, metaclass=ABCMeta):
    """Base class for all kinds of operators that act *locally*,
    i.e. only on a single degree of freedom.

    All :class:`LocalOperator` instances have a fixed associated identifier
    (symbol) that is used when printing that operator. A custom identifier can
    be useed through the associated
    :class:`~qnet.algebra.hilbert_space_algebra.LocalSpace`'s
    `local_identifiers` parameter. For example::

        >>> from qnet.printing import ascii
        >>> from qnet.algebra.operator_algebra import Destroy
        >>> hs1_custom = LocalSpace(1, local_identifiers={'Destroy': 'b'})
        >>> b = Destroy(hs=hs1_custom)
        >>> ascii(b)
        'b^(1)'
    """

    _simplifications = [implied_local_space(keys=['hs', ]), ]

    _identifier = None  # must be overridden by subclasses!
    _dagger = False  # do representations include a dagger?
    _nargs = 0  # number of arguments
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9().+-]+)?$')

    def __init__(self, *args, hs):
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        if not isinstance(hs, LocalSpace):
            raise TypeError("hs must be a LocalSpace")
        self._hs = hs
        if self._identifier is None:
            raise TypeError(
                r"Can't instantiate abstract class %s with undefined "
                r"_identifier" % self.__class__.__name__)
        if len(args) != self._nargs:
            raise ValueError("expected %d arguments, gotten %d"
                             % (self._nargs, len(args)))
        args_vals = []
        for arg in args:
            try:
                args_vals.append(float(arg))
            except (TypeError, ValueError):
                args_vals.append("~%s" % str(arg))
        self._order_key = KeyTuple(
            [self.__class__.__name__, 1.0] + args_vals)
        super().__init__(*args, hs=hs)

    @property
    def space(self):
        """Hilbert space of the operator
        (:class:`~qdyn.algebra.hilbert_space_algebra.LocalSpace` instance)"""
        return self._hs

    @property
    def args(self):
        """The positional arguments used for instantiating the operator"""
        return tuple()

    @property
    def kwargs(self):
        """The keyword arguments used for instantiating the operator"""
        return OrderedDict([('hs', self._hs)])

    @property
    def identifier(self):
        """The identifier (symbol) that is used when printing the operator.

        A custom identifier can
        be used through the associated
        :class:`~qnet.algebra.hilbert_space_algebra.LocalSpace`'s
        `local_identifiers` parameter. For example::

            >>> from qnet.algebra.operator_algebra import Destroy
            >>> a = Destroy(hs=1)
            >>> a.identifier
            'a'
            >>> hs1_custom = LocalSpace(1, local_identifiers={'Destroy': 'b'})
            >>> b = Destroy(hs=hs1_custom)
            >>> b.identifier
            'b'
            >>> from qnet.printing import ascii; ascii(b)
            'b^(1)'
        """

        identifier = self._hs._local_identifiers.get(
            self.__class__.__name__, self._identifier)
        if not self._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'"
                % (identifier, self._rx_identifier.pattern))
        return identifier

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def all_symbols(self):
        """Set of symbols used in the operator"""
        return set()


class OperatorOperation(Operator, Operation, metaclass=ABCMeta):
    """Base class for Operations acting only on Operator arguments, for when
    the Hilbert space of the operation result is the product space of the
    operands.
    """

    def __init__(self, *operands, **kwargs):
        op_spaces = [o.space for o in operands]
        self._space = ProductSpace.create(*op_spaces)
        self._order_key = KeyTuple([
            '~'+self.__class__.__name__, '__',
            1.0] + [op._order_key for op in operands])
        super().__init__(*operands, **kwargs)

    @property
    def space(self):
        """Hilbert space of the operation result"""
        return self._space

    def _simplify_scalar(self):
        return self.create(*[o.simplify_scalar() for o in self.operands])


class SingleOperatorOperation(Operator, Operation, metaclass=ABCMeta):
    """Base class for Operations that act on a single Operator"""

    def __init__(self, op, **kwargs):
        if isinstance(op, SCALAR_TYPES):
            op = op * IdentityOperator
        self._space = op.space
        self._order_key = op._order_key + KeyTuple((self.__class__.__name__,))
        super().__init__(op, **kwargs)

    @property
    def space(self):
        """Hilbert space for the operation result"""
        return self._space

    @property
    def operand(self):
        """The operator that the operation acts on"""
        return self.operands[0]

    def _series_expand(self, param, about, order):
        ope = self.operand.series_expand(param, about, order)
        return tuple(adjoint(opet) for opet in ope)


###############################################################################
# Operator algebra elements
###############################################################################


class OperatorSymbol(Operator, Expression):
    """Symbolic operator, parametrized by an identifier string and an
    associated Hilbert space.

    Args:
        identifier (str): Symbol identifier
        hs (HilbertSpace): associated Hilbert space (can be a
            :class:`~qnet.algebra.hilbert_space_algebra.ProductSpace`)
    """

    # Not a LocalOperator subclass because an OperatorSymbol may be defined for
    # a ProductSpace

    def __init__(self, identifier, *, hs):
        identifier = str(identifier)
        if not LocalOperator._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'" % (
                    identifier, LocalOperator._rx_identifier.pattern))
        self.identifier = identifier
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        elif isinstance(hs, tuple):
            hs = ProductSpace.create(*[LocalSpace(h) for h in hs])
        self._hs = hs
        self._order_key = KeyTuple((self.__class__.__name__, str(identifier),
                                    1.0))
        super().__init__(identifier, hs=hs)

    @property
    def args(self):
        """The positional arguments used for instantiating the operator"""
        return (self.identifier,)

    @property
    def kwargs(self):
        """The keyword arguments used for instantiating the operator"""
        return {'hs': self._hs}

    @property
    def space(self):
        """Hilbert space of the operator"""
        return self._hs

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def all_symbols(self):
        return {self}


@singleton_object
class IdentityOperator(Operator, Expression, metaclass=Singleton):
    """``IdentityOperator`` constant (singleton) object."""

    @property
    def space(self):
        """TrivialSpace"""
        return TrivialSpace

    @property
    def _order_key(self):
        return KeyTuple(('~~', self.__class__.__name__, 1.0))

    @property
    def args(self):
        return tuple()

    def _adjoint(self):
        return self

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def _pseudo_inverse(self):
        return self

    def __eq__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return other == 1
        return self is other

    def all_symbols(self):
        return set(())


II = IdentityOperator


@singleton_object
class ZeroOperator(Operator, Expression, metaclass=Singleton):
    """``ZeroOperator`` constant (singleton) object."""

    @property
    def space(self):
        """TrivialSpace"""
        return TrivialSpace

    @property
    def _order_key(self):
        return KeyTuple(('~~', self.__class__.__name__, 1.0))

    @property
    def args(self):
        return tuple()

    def _adjoint(self):
        return self

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + ((0,) * order)

    def _pseudo_inverse(self):
        return self

    def __eq__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return other == 0
        return self is other

    def all_symbols(self):
        return set(())


class Destroy(LocalOperator):
    """Bosonic annihilation operator acting on a particular
    :class:`LocalSpace` `hs`.

    It obeys the bosonic commutation relation::

        >>> Destroy(hs=1) * Create(hs=1) - Create(hs=1) * Destroy(hs=1)
        IdentityOperator
        >>> Destroy(hs=1) * Create(hs=2) - Create(hs=2) * Destroy(hs=1)
        ZeroOperator
    """
    _identifier = 'a'
    _dagger = False
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*$')

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    @property
    def identifier(self):
        """The identifier (symbols) that is used when printing the annihilation
        operator. This is identical to the identifier of :class:`Create`. A
        custom identifier for both :class:`Destroy` and :class:`Create` can be
        set through the `local_identifiers` parameter of the associated Hilbert
        space::

            >>> from qnet.algebra.operator_algebra import Create, Destroy
            >>> hs_custom = LocalSpace(0, local_identifiers={'Destroy': 'b'})
            >>> Create(hs=hs_custom).identifier
            'b'
            >>> Destroy(hs=hs_custom).identifier
            'b'
        """
        identifier = self._hs._local_identifiers.get(
            self.__class__.__name__, self._hs._local_identifiers.get(
                'Create', self._identifier))
        if not self._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'"
                % (identifier, self._rx_identifier.pattern))
        return identifier


class Create(LocalOperator):
    """Bosonic creation operator acting on a particular :class:`LocalSpace`
    `hs`. It is the adjoint of :class:`Destroy`.
    """
    _identifier = 'a'
    _dagger = True
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*$')

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    @property
    def identifier(self):
        """The identifier (symbols) that is used when printing the creation
        operator. This is identical to the identifier of :class:`Destroy`"""
        identifier = self._hs._local_identifiers.get(
            self.__class__.__name__, self._hs._local_identifiers.get(
                'Destroy', self._identifier))
        if not self._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'"
                % (identifier, self._rx_identifier.pattern))
        return identifier


class Jz(LocalOperator):
    """$\Op{J}_z$ is the $z$ component of a general spin operator acting
    on a particular :class:`LocalSpace` `hs` of freedom with well defined spin
    quantum number $s$.  It is Hermitian::

        >>> print(ascii(Jz(hs=1).adjoint()))
        J_z^(1)

    :class:`Jz`, :class:`Jplus` and :class:`Jminus` satisfy the angular
    momentum commutator algebra::

        >>> print(ascii((Jz(hs=1) * Jplus(hs=1) -
        ...              Jplus(hs=1)*Jz(hs=1)).expand()))
        J_+^(1)

        >>> print(ascii((Jz(hs=1) * Jminus(hs=1) -
        ...              Jminus(hs=1)*Jz(hs=1)).expand()))
        -J_-^(1)

        >>> print(ascii((Jplus(hs=1) * Jminus(hs=1)
        ...              - Jminus(hs=1)*Jplus(hs=1)).expand()))
        2 * J_z^(1)

    Printers should represent this operator with the default identifier::

        >>> Jz._identifier
        'J_z'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'J_z'

    def __init__(self, *, hs):
        super().__init__(hs=hs)


class Jplus(LocalOperator):
    """ $\Op{J}_{+} = \Op{J}_x + i \op{J}_y$ is the raising ladder operator
    of a general spin operator acting on a particular :class:`LocalSpace` `hs`
    with well defined spin quantum number $s$.  It's adjoint is the
    lowering operator::

        >>> print(ascii(Jplus(hs=1).adjoint()))
        J_-^(1)

    :class:`Jz`, :class:`Jplus` and :class:`Jminus` satisfy that angular
    momentum commutator algebra, see :class:`Jz`

    Printers should represent this operator with the default identifier::

        >>> Jplus._identifier
        'J_+'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'J_+'

    def __init__(self, *, hs):
        super().__init__(hs=hs)


class Jminus(LocalOperator):
    """$\Op{J}_{-} = \Op{J}_x - i \op{J}_y$ is lowering ladder operator of a
    general spin operator acting on a particular :class:`LocalSpace` `hs`
    with well defined spin quantum number $s$.  It's adjoint is the raising
    operator::

        >>> print(ascii(Jminus(hs=1).adjoint()))
        J_+^(1)

    :class:`Jz`, :class:`Jplus` and :class:`Jminus` satisfy that angular
    momentum commutator algebra, see :class:`Jz`.

    Printers should represent this operator with the default identifier::

        >>> Jminus._identifier
        'J_-'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'J_-'

    def __init__(self, *, hs):
        super().__init__(hs=hs)


class Phase(LocalOperator):
    r"""Unitary "phase" operator

    .. math::

        \op{P}_{\rm hs}(\phi) =
        \exp\left(i \phi \Op{a}_{\rm hs}^\dagger \Op{a}_{\rm hs}\right)

    where :math:`a_{\rm hs}` is the annihilation operator acting on the
    :class:`LocalSpace` `hs`.

    Printers should represent this operator with the default identifier::

        >>> Phase._identifier
        'Phase'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'Phase'
    _nargs = 1
    _rules = OrderedDict()  # see end of module
    _simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def __init__(self, phi, *, hs):
        self.phi = phi  #: Phase $\phi$
        super().__init__(phi, hs=hs)

    @property
    def args(self):
        r'''List of arguments of the operator, containing the phase $\phi$ as
        the only element'''
        return (self.phi,)

    def _diff(self, sym):
        raise NotImplementedError()

    def _adjoint(self):
        return Phase.create(-self.phi.conjugate(), hs=self.space)

    def _pseudo_inverse(self):
        return Phase.create(-self.phi, hs=self.space)

    def _simplify_scalar(self):
        return Phase.create(simplify_scalar(self.phi), hs=self.space)

    def all_symbols(self):
        return _scalar_free_symbols(self.space)


class Displace(LocalOperator):
    r"""Unitary coherent displacement operator

    .. math::

        \op{D}_{\rm hs}(\alpha) =
        \exp\left({\alpha \Op{a}_{\rm hs}^\dagger -
                   \alpha^* \Op{a}_{\rm hs}}\right)

    where :math:`\Op{a}_{\rm hs}` is the annihilation operator acting on the
    :class:`LocalSpace` `hs`.

    Printers should represent this operator with the default identifier::

        >>> Displace._identifier
        'D'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'D'
    _nargs = 1
    _rules = OrderedDict()  # see end of module
    _simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def __init__(self, alpha, *, hs):
        self.alpha = alpha  #: Displacement amplitude $\alpha$
        super().__init__(alpha, hs=hs)

    @property
    def args(self):
        r'''List of arguments of the operator, containing the displacement
        amplitude $\alpha$ as the only element'''
        return (self.alpha,)

    def _diff(self, sym):
        raise NotImplementedError()

    def _adjoint(self):
        return Displace.create(-self.alpha, hs=self.space)

    _pseudo_inverse = _adjoint

    def _simplify_scalar(self):
        return Displace.create(simplify_scalar(self.alpha), hs=self.space)

    def all_symbols(self):
        return _scalar_free_symbols(self.space)


class Squeeze(LocalOperator):
    r"""Unitary Squeezing operator

    .. math::

        \Op{S}_{\rm hs}(\eta) =
        \exp {\left( \frac{\eta}{2} {\Op{a}_{\rm hs}^\dagger}^2 -
                     \frac{\eta^*}{2} {\Op{a}_{\rm hs}}^2 \right)}

    where :math:`\Op{a}_{\rm hs}` is the annihilation operator acting on the
    :class:`LocalSpace` `hs`.

    Printers should represent this operator with the default identifier::

        >>> Squeeze._identifier
        'Squeeze'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = "Squeeze"
    _nargs = 1
    _rules = OrderedDict()  # see end of module
    _simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def __init__(self, eta, *, hs):
        self.eta = eta  #: sqeezing parameter $\eta$
        super().__init__(eta, hs=hs)

    @property
    def args(self):
        return (self.eta,)

    def _diff(self, sym):
        raise NotImplementedError()

    def _adjoint(self):
        return Squeeze(-self.eta, hs=self.space)

    _pseudo_inverse = _adjoint

    def _simplify_scalar(self):
        return Squeeze(simplify_scalar(self.eta), hs=self.space)

    def all_symbols(self):
        r'''List of arguments of the operator, containing the squeezing
        parameter $\eta$ as the only element'''
        return _scalar_free_symbols(self.space)


class LocalSigma(LocalOperator):
    r'''A local level flip operator operator acting on a particular
    :class:`LocalSpace` `hs`.

    .. math::

        \Op{\sigma}_{jk}^{\rm hs} =
        \left| j\right\rangle_{\rm hs} \left \langle k \right |_{\rm hs}

    For $j=k$ this becomes a projector $\Op{P}_k$ onto the eigenstate
    $\ket{k}$; see :class:`LocalProjector`.

    Args:
        j (int or str): The label or index identifying $\ket{j}$
        k (int or str):  The label or index identifying $\ket{k}$
        hs (HilbertSpace): The Hilbert space on which the operator acts

    Note:
        The parameters `j` or `k` may be an integer or a string. A string
        refers to the label of an eigenstate in the basis of `hs`, which needs
        to be set. An integer refers to the (zero-based) index of eigenstate of
        the Hilbert space. This works if `hs` has an unknown dimension.

    Raises:
        ValueError: If `j` or `k` are invalid value for the given `hs`

    Printers should represent this operator either in braket notation, or using
    the operator identifier

        >>> LocalSigma(0, 0, hs=0).identifier
        'sigma'
    '''
    _identifier = "sigma"
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*$')
    _nargs = 2
    _rules = OrderedDict()  # see end of module
    _simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def __init__(self, j, k, *, hs):
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        if not isinstance(hs, LocalSpace):
            raise TypeError("hs must be a LocalSpace")
        for ind_jk in range(2):
            jk = j if ind_jk == 0 else k
            hs._check_basis_label_type(jk)
            if isinstance(jk, str):
                if not hs.has_basis:
                    raise ValueError(
                        "Invalid to give label %s for Hilbert space %s that "
                        "has no basis" % (jk, hs))
            elif isinstance(jk, int):
                if jk < 0:
                    raise ValueError("Index j/k=%s must be >= 0" % jk)
                if hs.has_basis:
                    if jk >= hs.dimension:
                        raise ValueError(
                            "Index j/k=%s must be < the Hilbert space "
                            "dimension %d" % (jk, hs.dimension))
                    if ind_jk == 0:
                        j = hs.basis_labels[jk]
                    else:
                        k = hs.basis_labels[jk]
            elif isinstance(jk, SymbolicLabelBase):
                pass  # use j, k as-is
            else:
                # Interal error: mismatch with hs._basis_label_types
                raise NotImplementedError()
        self.j = j  #: label/index of eigenstate  $\ket{j}$
        self.k = k  #: label/index of eigenstate  $\ket{k}$
        super().__init__(j, k, hs=hs)

    @property
    def args(self):
        """The two eigenstate labels `j` and `k` that the operator connects"""
        return self.j, self.k

    @property
    def index_j(self):
        """Index `j` or (zero-based) index of the label `j` in the basis"""
        if isinstance(self.j, (int, SymbolicLabelBase)):
            return self.j
        else:
            return self.space.basis_labels.index(self.j)

    @property
    def index_k(self):
        """Index `k` or (zero-based) index of the label `k` in the basis"""
        if isinstance(self.k, (int, SymbolicLabelBase)):
            return self.k
        else:
            return self.space.basis_labels.index(self.k)

    def raise_jk(self, j_incr=0, k_incr=0):
        r'''Return a new :class:`LocalSigma` instance with incremented `j`,
        `k`, on the same Hilbert space:

        .. math::

            \Op{\sigma}_{jk}^{\rm hs} \rightarrow \Op{\sigma}_{j'k'}^{\rm hs}

        This is the result of multiplying $\Op{\sigma}_{jk}^{\rm hs}$
        with any raising or lowering operators.

        If $j'$ or $k'$ are outside the Hilbert space ${\rm hs}$, the result is
        the :obj:`ZeroOperator` .

        Args:
            j_incr (int): The increment between labels $j$ and $j'$
            k_incr (int): The increment between labels $k$ and $k'$. Both
                increments may be negative.
        '''
        try:
            if isinstance(self.j, int):
                new_j = self.j + j_incr
            else:  # str
                new_j = self.space.next_basis_label_or_index(self.j, j_incr)
            if isinstance(self.k, int):
                new_k = self.k + k_incr
            else:  # str or SymbolicLabelBase
                new_k = self.space.next_basis_label_or_index(self.k, k_incr)
            return LocalSigma.create(new_j, new_k, hs=self.space)
        except (IndexError, ValueError):
            return ZeroOperator


class LocalProjector(LocalSigma):
    """A projector onto a specific level.

    Args:
        j (int or str): The label or index identifying the state onto which
            is projected
        hs (HilbertSpace): The Hilbert space on which the operator acts
    """
    _identifier = "Pi"
    _nargs = 2  # must be 2 because that's how we call super().__init__

    def __init__(self, j, *, hs):
        super().__init__(j=j, k=j, hs=hs)

    @property
    def args(self):
        """One-element tuple containing eigenstate label `j` that the projector
        projects onto"""
        return (self.j,)


###############################################################################
# Algebra Operations
###############################################################################


class OperatorPlus(OperatorOperation):
    """A sum of Operators

    Args:
        operands (list): Operator summands
    """
    neutral_element = ZeroOperator
    _binary_rules = OrderedDict()
    _simplifications = [assoc, scalars_to_op, orderby, filter_neutral,
                        match_replace_binary]

    order_key = FullCommutativeHSOrder

    def _expand(self):
        summands = [o.expand() for o in self.operands]
        return OperatorPlus.create(*summands)

    def _series_expand(self, param, about, order):
        tuples = (o.series_expand(param, about, order) for o in self.operands)
        res = (OperatorPlus.create(*tels) for tels in zip(*tuples))
        return res

    def _diff(self, sym):
        return sum([o._diff(sym) for o in self.operands], ZeroOperator)


class OperatorTimes(OperatorOperation):
    """A product of Operators that serves both as a product within a Hilbert
    space as well as a tensor product.

    Args:
        operands (list): Operator factors
    """

    neutral_element = IdentityOperator
    _binary_rules = OrderedDict()  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary]

    order_key = DisjunctCommutativeHSOrder

    def factor_for_space(self, spc):
        if spc == TrivialSpace:
            ops_on_spc = [o for o in self.operands
                          if o.space is TrivialSpace]
            ops_not_on_spc = [o for o in self.operands
                              if o.space > TrivialSpace]
        else:
            ops_on_spc = [o for o in self.operands
                          if (o.space & spc) > TrivialSpace]
            ops_not_on_spc = [o for o in self.operands
                              if (o.space & spc) is TrivialSpace]
        return (OperatorTimes.create(*ops_on_spc),
                OperatorTimes.create(*ops_not_on_spc))

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, OperatorPlus) else (eo,)
                        for eo in eops]
        # iterate over a cartesian product of all factor summands, form product
        # of each tuple and sum over result
        summands = []
        for combo in cartesian_product(*eopssummands):
            summand = OperatorTimes.create(*combo)
            summands.append(summand)
        ret = OperatorPlus.create(*summands)
        if isinstance(ret, OperatorPlus):
            return ret.expand()
        else:
            return ret

    def _series_expand(self, param, about, order):
        assert len(self.operands) > 1
        cfirst = self.operands[0].series_expand(param, about, order)
        crest = OperatorTimes.create(*self.operands[1:]).series_expand(
            param, about, order)
        res = []
        for n in range(order + 1):
            summands = [cfirst[k] * crest[n - k] for k in range(n + 1)]
            res.append(OperatorPlus.create(*summands))
        return tuple(res)

    def _diff(self, sym):
        assert len(self.operands) > 1
        first = self.operands[0]
        rest = OperatorTimes.create(*self.operands[1:])
        return first._diff(sym) * rest + first * rest._diff(sym)


class ScalarTimesOperator(Operator, ScalarTimesExpression):
    """Multiply an operator by a scalar coefficient.

    Args:
        coeff (SCALAR_TYPES): coefficient
        term (Operator): operator
    """
    _rules = OrderedDict()
    _simplifications = [match_replace, ]

    @staticmethod
    def has_minus_prefactor(c):
        """
        For a scalar object c, determine whether it is prepended by a "-" sign.
        """
        cs = str(c).strip()
        return cs[0] == "-"

    @property
    def _order_key(self):
        t = self.term._order_key
        try:
            c = abs(float(self.coeff))  # smallest coefficients first
        except (ValueError, TypeError):
            c = float('inf')
        return KeyTuple(t[:2] + (c,) + t[3:] + (str(self.coeff),))

    @property
    def space(self):
        return self.operands[1].space

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, OperatorPlus):
            summands = [c * eto for eto in et.operands]
            return OperatorPlus.create(*summands)
        return c * et

    def _series_expand(self, param, about, order):
        te = self.term.series_expand(param, about, order)
        if isinstance(self.coeff, SympyBasic):
            if about != 0:
                c = self.coeff.subs({param: about + param})
            else:
                c = self.coeff

            series = sympy_series(c, x=param, x0=0, n=None)
            ce = []
            next_order = 0
            for term in series:
                c, o = term.as_coeff_exponent(param)
                if o < 0:
                    raise ValueError("{} is singular at expansion point "
                                     "{} = {}.".format(self, param, about))
                if o > order:
                    break
                ce.extend([0] * (o - next_order))
                ce.append(c)
                next_order = o + 1
            ce.extend([0] * (order + 1 - next_order))

            res = []
            for n in range(order + 1):
                summands = [ce[k] * te[n - k] for k in range(n + 1)]
                res.append(OperatorPlus.create(*summands))
            return tuple(res)
        else:
            return tuple(self.coeff * tek for tek in te)

    def _diff(self, sym):
        c, t = self.operands
        cd = c.diff(sym) if isinstance(c, SympyBasic) else 0
        return cd * t + c * t._diff(sym)

    def _pseudo_inverse(self):
        c, t = self.operands
        return t.pseudo_inverse() / c

    def __eq__(self, other):
        if (self.term is IdentityOperator and
                isinstance(other, SCALAR_TYPES)):
            return self.coeff == other
        return super().__eq__(other)

    # When overloading __eq__ it can mess up the hashability unless we
    # explicitly also overload  __hash__, very subtle python3 bug, first found
    # by David Richie.
    def __hash__(self):
        return super().__hash__()

    def _simplify_scalar(self):
        coeff, term = self.operands
        return simplify_scalar(coeff) * term.simplify_scalar()


class Commutator(OperatorOperation):
    r'''Commutator of two operators

    .. math::

        [\Op{A}, \Op{B}] = \Op{A}\Op{B} - \Op{A}\Op{B}

    '''
    _rules = OrderedDict()
    _simplifications = [
        scalars_to_op, disjunct_hs_zero, commutator_order, match_replace]
    # TODO: doit method

    order_key = FullCommutativeHSOrder

    # commutator_order makes FullCommutativeHSOrder anti-commutative

    def __init__(self, A, B):
        self._hs = A.space * B.space
        super().__init__(A, B)

    @property
    def A(self):
        """Left side of the commutator"""
        return self.operands[0]

    @property
    def B(self):
        """Left side of the commutator"""
        return self.operands[1]

    def _expand(self):
        A = self.A.expand()
        B = self.B.expand()
        if isinstance(A, OperatorPlus):
            A_summands = A.operands
        else:
            A_summands = (A,)
        if isinstance(B, OperatorPlus):
            B_summands = B.operands
        else:
            B_summands = (B,)
        summands = []
        for combo in cartesian_product(A_summands, B_summands):
            summands.append(Commutator.create(*combo))
        return OperatorPlus.create(*summands)

    def _series_expand(self, param, about, order):
        A_series = self.A.series_expand(param, about, order)
        B_series = self.B.series_expand(param, about, order)
        res = []
        for n in range(order + 1):
            summands = [self.create(A_series[k], B_series[n - k])
                        for k in range(n + 1)]
            res.append(OperatorPlus.create(*summands))
        return tuple(res)

    def _diff(self, sym):
        return (self.__class__(self.A.diff(sym), self.B) +
                self.__class__(self.A, self.B.diff(sym)))


class OperatorTrace(SingleOperatorOperation):
    r'''Take the (partial) trace of an operator `op` ($\Op{O}) over the degrees
    of freedom of a Hilbert space `over_space` ($\mathcal{H}$):

    .. math::

        {\rm Tr}_{\mathcal{H}} \Op{O}

    Args:
        over_space (HilbertSpace): The degrees of freedom to trace over
        op (Opwerator): The operator to take the trace of.
    '''
    _rules = OrderedDict()  # see end of module
    _simplifications = [
        scalars_to_op, implied_local_space(keys=['over_space', ]),
        match_replace, ]

    def __init__(self, op, *, over_space):
        if isinstance(over_space, (int, str)):
            over_space = LocalSpace(over_space)
        assert isinstance(over_space, HilbertSpace)
        self._over_space = over_space
        super().__init__(op, over_space=over_space)
        self._space = None
        self._order_key = (op._order_key +
                           KeyTuple((self.__class__.__name__,
                                     over_space._order_key)))

    @property
    def kwargs(self):
        return {'over_space': self._over_space}

    @property
    def operand(self):
        return self.operands[0]

    @property
    def space(self):
        if self._space is None:
            return self.operands[0].space / self._over_space
        return self._space

    def _expand(self):
        s = self._over_space
        o = self.operand
        return OperatorTrace.create(o.expand(), over_space=s)

    def _series_expand(self, param, about, order):
        ope = self.operand.series_expand(param, about, order)
        return tuple(OperatorTrace.create(opet, over_space=self._over_space)
                     for opet in ope)

    def all_symbols(self):
        return self.operand.all_symbols()

    def _diff(self, sym):
        s = self._over_space
        o = self.operand
        return OperatorTrace.create(o._diff(sym), over_space=s)


class Adjoint(SingleOperatorOperation):
    """The symbolic Adjoint of an operator.

    :param op: The operator to take the adjoint of.
    :type op: Operator
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [
        scalars_to_op, match_replace, delegate_to_method('_adjoint')]

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, OperatorPlus):
            summands = [adjoint(eoo) for eoo in eo.operands]
            return OperatorPlus.create(*summands)
        return eo.adjoint()

    def _pseudo_inverse(self):
        return self.operand.pseudo_inverse().adjoint()

    def _diff(self, sym):
        return Adjoint.create(self.operands[0]._diff(sym))


class OperatorPlusMinusCC(SingleOperatorOperation):
    """An operator plus or minus its complex conjugate"""

    def __init__(self, op, *, sign=+1):
        self._sign = sign
        super().__init__(op, sign=sign)

    @property
    def kwargs(self):
        if self._sign > 0:
            return {'sign': +1, }
        else:
            return {'sign': -1, }

    @property
    def minimal_kwargs(self):
        if self._sign == +1:
            return {}
        else:
            return self.kwargs

    def _expand(self):
        return self

    def _pseudo_inverse(self):
        return OperatorPlusMinusCC(self.operand.pseudo_inverse(), self._sign)

    def _diff(self, sym):
        return OperatorPlusMinusCC(self.operands._diff(sym), self._sign)


class PseudoInverse(SingleOperatorOperation):
    r"""The symbolic pseudo-inverse :math:`X^+` of an operator :math:`X`. It is
    defined via the relationship

    .. math::

        X X^+ X =  X  \\
        X^+ X X^+ =  X^+  \\
        (X^+ X)^\dagger = X^+ X  \\
        (X X^+)^\dagger = X X^+

    :param X: The operator to take the adjoint of.
    :type X: Operator
    """
    _rules = OrderedDict()  # see end of module
    _delegate_to_method = (ScalarTimesOperator, Squeeze, Displace,
                           ZeroOperator.__class__, IdentityOperator.__class__)
    _simplifications = [
        scalars_to_op, match_replace, delegate_to_method('_pseudo_inverse')]

    def _expand(self):
        return self

    def _pseudo_inverse(self):
        return self.operand


class NullSpaceProjector(SingleOperatorOperation):
    r"""Returns a projection operator :math:`\mathcal{P}_{{\rm Ker} X}` that
    projects onto the nullspace of its operand

    .. math::

        X \mathcal{P}_{{\rm Ker} X} = 0 \Leftrightarrow  X (1 - \mathcal{P}_{{\rm Ker} X}) = X\\
        \mathcal{P}_{{\rm Ker} X}^\dagger = \mathcal{P}_{{\rm Ker} X} = \mathcal{P}_{{\rm Ker} X}^2

    :param X: Operator argument
    :type X: Operator
    """

    _rules = OrderedDict()  # see end of module
    _simplifications = [scalars_to_op, match_replace, ]

    def _expand(self):
        return self


class OperatorIndexedSum(IndexedSum, SingleOperatorOperation):
    # Order of superclasses is important for proper mro for __add__ etc.
    # (we're using cooperative inheritance from both superclasses,
    # cf. https://stackoverflow.com/q/47804919)

    # TODO: documentation

    _rules = OrderedDict()  # see end of module TODO
    _simplifications = [
        assoc_indexed, scalars_to_op, indexed_sum_over_const,
        indexed_sum_over_kronecker, match_replace, ]

    @property
    def space(self):
        return self.term.space

    def _expand(self):
        return self.__class__.create(self.term.expand(), *self.ranges)

    def _series_expand(self, param, about, order):
        raise NotImplementedError()

    def _adjoint(self):
        return self.__class__.create(self.term._adjoint(), *self.ranges)


###############################################################################
# Constructor Routines
###############################################################################


tr = OperatorTrace.create


def X(local_space, states=("h", "g")):
    r"""Pauli-type X-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or str
    :return: Local X-operator as a linear combination of :class:`LocalSigma`
    :rtype: Operator
    """
    h, g = states  # TODO: default should be 0, 1
    return (
        LocalSigma.create(h, g, hs=local_space) +
        LocalSigma.create(g, h, hs=local_space))


def Y(local_space, states=("h", "g")):
    r""" Pauli-type Y-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or str
    :return: Local Y-operator as a linear combination of :class:`LocalSigma`
    :rtype: Operator
    """
    h, g = states
    return I * (-LocalSigma.create(h, g, hs=local_space) +
                LocalSigma.create(g, h, hs=local_space))


def Z(local_space, states=("h", "g")):
    r"""Pauli-type Z-operator

    :param local_space: Associated Hilbert space.
    :type local_space: LocalSpace
    :param states: The qubit state labels for the basis states :math:`\left\{|0\rangle, |1\rangle \right\}`, where :math:`Z|0\rangle = +|0\rangle`, default = ``('h', 'g')``.
    :type states: tuple with two elements of type int or str
    :return: Local Z-operator as a linear combination of :class:`LocalSigma`
    :rtype: Operator
    """
    h, g = states
    return (LocalProjector(h, hs=local_space) -
            LocalProjector(g, hs=local_space))


###############################################################################
# Auxilliary routines
###############################################################################


def factor_for_trace(ls, op):
    r'''Given a :class:`LocalSpace` `ls` to take the partial trace over and an
    operator `op`, factor the trace such that operators acting on disjoint
    degrees of freedom are pulled out of the trace. If the operator acts
    trivially on ls the trace yields only a pre-factor equal to the dimension
    of ls. If there are LocalSigma operators among a product, the trace's
    cyclical property is used to move to sandwich the full product by
    :py:class:`LocalSigma` operators:

    .. math::

        {\rm Tr} A \sigma_{jk} B = {\rm Tr} \sigma_{jk} B A \sigma_{jj}

    :param ls: Degree of Freedom to trace over
    :type ls: HilbertSpace
    :param op: Operator to take the trace of
    :type op: Operator
    :return: The (partial) trace over the operator's spc-degrees of freedom
    :rtype: Operator
    '''
    if op.space == ls:
        if isinstance(op, OperatorTimes):
            pull_out = [o for o in op.operands if o.space is TrivialSpace]
            rest = [o for o in op.operands if o.space is not TrivialSpace]
            if pull_out:
                return (OperatorTimes.create(*pull_out) *
                        OperatorTrace.create(OperatorTimes.create(*rest),
                                             over_space=ls))
        raise CannotSimplify()
    if ls & op.space == TrivialSpace:
        return ls.dimension * op
    if ls < op.space and isinstance(op, OperatorTimes):
        pull_out = [o for o in op.operands if (o.space & ls) == TrivialSpace]

        rest = [o for o in op.operands if (o.space & ls) != TrivialSpace]
        if (not isinstance(rest[0], LocalSigma) or
                not isinstance(rest[-1], LocalSigma)):
            found_ls = False
            for j, r in enumerate(rest):
                if isinstance(r, LocalSigma):
                    found_ls = True
                    break
            if found_ls:
                m = r.j
                rest = rest[j:] + rest[:j] + [LocalSigma.create(m, m, hs=ls), ]
        if not rest:
            rest = [IdentityOperator]
        if len(pull_out):
            return (OperatorTimes.create(*pull_out) *
                    OperatorTrace.create(OperatorTimes.create(*rest),
                                         over_space=ls))
    raise CannotSimplify()


def decompose_space(H, A):
    """Simplifies OperatorTrace expressions over tensor-product spaces by
    turning it into iterated partial traces.

    :param H: The full space.
    :type H: ProductSpace
    :type A: Operator
    :return: Iterative partial trace expression
    :rtype: Operator
    """
    return OperatorTrace.create(
        OperatorTrace.create(A, over_space=H.operands[-1]),
        over_space=ProductSpace.create(*H.operands[:-1]))


def get_coeffs(expr, expand=False, epsilon=0.):
    """Create a dictionary with all Operator terms of the expression
    (understood as a sum) as keys and their coefficients as values.

    The returned object is a defaultdict that return 0. if a term/key
    doesn't exist.
    :param expr: The operator expression to get all coefficients from.
    :param expand: Whether to expand the expression distributively.
    :param epsilon: If non-zero, drop all Operators with coefficients that have absolute value less than epsilon.
    :return: A dictionary of {op1: coeff1, op2: coeff2, ...}
    :rtype: dict
    """
    if expand:
        expr = expr.expand()
    ret = defaultdict(int)
    operands = expr.operands if isinstance(expr, OperatorPlus) else [expr]
    for e in operands:
        c, t = _coeff_term(e)
        try:
            if abs(complex(c)) < epsilon:
                continue
        except TypeError:
            pass
        ret[t] += c
    return ret


def space(obj):
    """Gives the associated HilbertSpace with an object. Also works for
    `SCALAR_TYPES` (returning
    :class:`~qnet.algebra.hilbert_space_algebra.TrivialSpace`)
    """
    try:
        return obj.space
    except AttributeError:
        if isinstance(obj, SCALAR_TYPES):
            return TrivialSpace
        raise ValueError(str(obj))


def simplify_scalar(s):
    """Simplify all occurences of scalar expressions in s

    :param s: The expression to simplify.
    :type s: Expression or SympyBasic
    :return: The simplified version.
    :rtype: Expression or SympyBasic
    """
    try:
        return s.simplify_scalar()
    except AttributeError:
        pass
    if isinstance(s, SympyBasic):
        return s.simplify()
    return s


def _coeff_term(op):
    if isinstance(op, ScalarTimesOperator):
        return op.coeff, op.term
    elif isinstance(op, SCALAR_TYPES):
        if op == 0:
            return 0, ZeroOperator
        else:
            return op, IdentityOperator
    else:
        return 1, op


def factor_coeff(cls, ops, kwargs):
    """Factor out coefficients of all factors."""
    coeffs, nops = zip(*map(_coeff_term, ops))
    coeff = 1
    for c in coeffs:
        coeff *= c
    if coeff == 1:
        return nops, coeffs
    else:
        return coeff * cls.create(*nops, **kwargs)


def adjoint(obj):
    """Return the adjoint of an obj."""
    try:
        return obj.adjoint()
    except AttributeError:
        return obj.conjugate()


def Jpjmcoeff(ls, m, shift=False):
    r'''Eigenvalue of the $\Op{J}_{+}$ (:class:`Jplus`) operator, as a Sympy
    expression.

    .. math::

        \Op{J}_{+} \ket{s, m} = \sqrt{s (s+1) - m (m+1)} \ket{s, m}

    where the multiplicity $s$ is implied by the size of the Hilbert space
    `ls`: there are $2s+1$ eigenstates with $m = -s, -s+1, \dots, s$.

    Args:
        ls (LocalSpace): The Hilbert space in which the $\Op{J}_{+}$ operator
            acts.
        m (Any): If str, the label of the basis state of `hs` to which
            the operator is applied. If integer together with ``shift=True``,
            the zero-based index of the basis state. Otherwise, directly the
            quantum number $m$.
        shift (bool): If True for a integer value of `m`, treat `m` as the
            zero-based index of the basis state (i.e., shift `m` down by $s$ to
            obtain the quantum number $m$)
    '''
    try:
        n = ls.dimension
        s = sympify(n - 1) / 2
        assert n == int(2 * s + 1)
        if isinstance(m, str):
            m = ls.basis.index(m) - s  # m is now Sympy expression
        elif isinstance(m, int):
            if shift:
                assert 0 <= m < n
                m = m - s
        return sqrt(s * (s + 1) - m * (m + 1))
    except BasisNotSetError:
        raise CannotSimplify()


def Jzjmcoeff(ls, m, shift):
    r'''Eigenvalue of the $\Op{J}_z$ (:class:`Jz`) operator, as a Sympy
    expression.

    .. math::

        \Op{J}_{z} \ket{s, m} = m \ket{s, m}

    See also :func:`Jpjmcoeff`.
    '''
    try:
        n = ls.dimension
        s = sympify(n - 1) / 2
        assert n == int(2 * s + 1)
        if isinstance(m, str):
            return ls.basis.index(m) - s
        elif isinstance(m, int):
            if shift:
                assert 0 <= m < n
                return m - s
        else:
            return sympify(m)
    except BasisNotSetError:
        raise CannotSimplify()


def Jmjmcoeff(ls, m, shift):
    r'''Eigenvalue of the $\Op{J}_{-}$ (:class:`Jminus`) operator, as a Sympy
    expression

    .. math::

        \Op{J}_{-} \ket{s, m} = \sqrt{s (s+1) - m (m-1)} \ket{s, m}

    See also :func:`Jpjmcoeff`.
    '''
    try:
        n = ls.dimension
        s = sympify(n - 1) / 2
        assert n == int(2 * s + 1)
        if isinstance(m, str):
            m = ls.basis.index(m) - s  # m is now Sympy expression
        elif isinstance(m, int):
            if shift:
                assert 0 <= m < n
                m = m - s
        return sqrt(s * (s + 1) - m * (m - 1))
    except BasisNotSetError:
        raise CannotSimplify()


###############################################################################
# Extra ("manual") simplifications
###############################################################################


def _combine_operator_p_cc(A, B):
    if B.adjoint() == A:
        return OperatorPlusMinusCC(A, sign=+1)
    else:
        raise CannotSimplify


def _combine_operator_m_cc(A, B):
    if B.adjoint() == A:
        return OperatorPlusMinusCC(A, sign=-1)
    else:
        raise CannotSimplify


def _scal_combine_operator_pm_cc(c, A, d, B):
    if B.adjoint() == A:
        if c == d:
            return c * OperatorPlusMinusCC(A, sign=+1)
        elif c == -d:
            return c * OperatorPlusMinusCC(A, sign=-1)
    raise CannotSimplify


def create_operator_pm_cc():
    """Return a list of rules that can be used in an
    :func:`~qnet.algebra.abstract_algebra.extra_binary_rules`
    context for :class:`OperatorPlus` in order to combine suitable terms into a
    :class:`OperatorPlusMinusCC` instance::

        >>> A = OperatorSymbol('A', hs=1)
        >>> sum = A + A.dag()
        >>> from qnet.algebra.abstract_algebra import extra_binary_rules
        >>> with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
        ...     sum2 = sum.simplify()
        >>> print(ascii(sum2))
        A^(1) + c.c.

    The inverse is done through :func:`expand_operator_pm_cc`::

        >>> print(ascii(sum2.simplify(rules=expand_operator_pm_cc())))
        A^(1) + A^(1)H
    """
    A = wc("A", head=Operator)
    B = wc("B", head=Operator)
    c = wc("c", head=SCALAR_TYPES)
    d = wc("d", head=SCALAR_TYPES)
    return [
        ('pmCC1', (
            pattern_head(A, B),
            _combine_operator_p_cc)),
        ('pmCC2', (
            pattern_head(pattern(ScalarTimesOperator, -1, B), A),
            _combine_operator_m_cc)),
        ('pmCC3', (
            pattern_head(
                pattern(ScalarTimesOperator, c, A),
                pattern(ScalarTimesOperator, d, B)),
            _scal_combine_operator_pm_cc)),
    ]


def expand_operator_pm_cc():
    """Return a list of rules that can be used in `simplify` to expand
    instances of :class:`OperatorPlusMinusCC`

    Inverse of :func:`create_operator_pm_cc`.
    """
    A = wc("A", head=Operator)
    return OrderedDict([
        ('pmCCexpand1', (
            pattern(OperatorPlusMinusCC, A, sign=+1),
            lambda A: A + A.dag())),
        ('pmCCexpand2', (
            pattern(OperatorPlusMinusCC, A, sign=-1),
            lambda A: A - A.dag())),
    ])


###############################################################################
# Algebraic rules
###############################################################################


def _algebraic_rules():
    """Set the default algebraic rules for the operations defined in this
    module"""
    PseudoInverse._delegate_to_method += (PseudoInverse,)

    u = wc("u", head=SCALAR_TYPES)
    v = wc("v", head=SCALAR_TYPES)

    n = wc("n", head=(int, str, SymbolicLabelBase))
    m = wc("m", head=(int, str, SymbolicLabelBase))

    A = wc("A", head=Operator)
    B = wc("B", head=Operator)

    A_plus = wc("A", head=OperatorPlus)
    A_times = wc("A", head=OperatorTimes)

    ls = wc("ls", head=LocalSpace)
    h1 = wc("h1", head=HilbertSpace)
    H_ProductSpace = wc("H", head=ProductSpace)

    localsigma = wc(
        'localsigma', head=(LocalSigma, LocalProjector), kwargs={'hs': ls})

    ra = wc("ra", head=(int, str, SymbolicLabelBase))
    rb = wc("rb", head=(int, str, SymbolicLabelBase))
    rc = wc("rc", head=(int, str, SymbolicLabelBase))
    rd = wc("rd", head=(int, str, SymbolicLabelBase))

    indranges__ = wc("indranges__", head=IndexRangeBase)

    ScalarTimesOperator._rules.update(check_rules_dict([
        ('1A', (
            pattern_head(1, A),
            lambda A: A)),
        ('0A', (
            pattern_head(0, A),
            lambda A: ZeroOperator)),
        ('u0', (
            pattern_head(u, ZeroOperator),
            lambda u: ZeroOperator)),
        ('assoc_coeff', (
            pattern_head(u, pattern(ScalarTimesOperator, v, A)),
            lambda u, v, A: (u * v) * A)),
        ('negsum', (
            pattern_head(-1, A_plus),
            lambda A: OperatorPlus.create(*[-1 * op for op in A.args]))),
    ]))

    OperatorPlus._binary_rules.update(check_rules_dict([
        ('upv', (
            pattern_head(
                pattern(ScalarTimesOperator, u, A),
                pattern(ScalarTimesOperator, v, A)),
            lambda u, v, A: (u + v) * A)),
        ('up1', (
            pattern_head(pattern(ScalarTimesOperator, u, A), A),
            lambda u, A: (u + 1) * A)),
        ('1pv', (
            pattern_head(A, pattern(ScalarTimesOperator, v, A)),
            lambda v, A: (1 + v) * A)),
        ('2A', (
            pattern_head(A, A),
            lambda A: 2 * A)),
    ]))

    OperatorTimes._binary_rules.update(check_rules_dict([
        ('uAB', (
            pattern_head(pattern(ScalarTimesOperator, u, A), B),
            lambda u, A, B: u * (A * B))),

        ('zero1', (
            pattern_head(ZeroOperator, B),
            lambda B: ZeroOperator)),
        ('zero2', (
            pattern_head(A, ZeroOperator),
            lambda A: ZeroOperator)),

        ('coeff', (
            pattern_head(A, pattern(ScalarTimesOperator, u, B)),
            lambda A, u, B: u * (A * B))),

        ('sig', (
            pattern_head(
                pattern(LocalSigma, ra, rb, hs=ls),
                pattern(LocalSigma, rc, rd, hs=ls)),
            lambda ls, ra, rb, rc, rd: (
                KroneckerDelta(rb, rc) * LocalSigma.create(ra, rd, hs=ls)))),
        ('sigproj', (
            pattern_head(
                pattern(LocalSigma, ra, rb, hs=ls),
                pattern(LocalProjector, rc, hs=ls)),
            lambda ls, ra, rb, rc: (
                KroneckerDelta(rb, rc) * LocalSigma.create(ra, rc, hs=ls)))),
        ('projsig', (
            pattern_head(
                pattern(LocalProjector, ra, hs=ls),
                pattern(LocalSigma, rc, rd, hs=ls)),
            lambda ls, ra, rc, rd: (
                KroneckerDelta(ra, rc) * LocalSigma.create(ra, rd, hs=ls)))),
        ('projproj', (
            pattern_head(
                pattern(LocalProjector, ra, hs=ls),
                pattern(LocalProjector, rc, hs=ls)),
            lambda ls, ra, rc: (
                KroneckerDelta(ra, rc) * LocalProjector(ra, hs=ls)))),

        # Harmonic oscillator rules
        ('hamos1', (
            pattern_head(pattern(Create, hs=ls), localsigma),
            lambda ls, localsigma:
                sqrt(localsigma.index_j + 1) * localsigma.raise_jk(j_incr=1))),

        ('hamos2', (
            pattern_head(pattern(Destroy, hs=ls), localsigma),
            lambda ls, localsigma:
                sqrt(localsigma.index_j) * localsigma.raise_jk(j_incr=-1))),

        ('hamos3', (
            pattern_head(localsigma, pattern(Destroy, hs=ls)),
            lambda ls, localsigma:
                sqrt(localsigma.index_k + 1) * localsigma.raise_jk(k_incr=1))),

        ('hamos4', (
            pattern_head(localsigma, pattern(Create, hs=ls)),
            lambda ls, localsigma:
                sqrt(localsigma.index_k) * localsigma.raise_jk(k_incr=-1))),

        # Normal ordering for harmonic oscillator <=> all a^* to the left, a to
        # the right.
        ('hamosord', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Create, hs=ls)),
            lambda ls: IdentityOperator + Create(hs=ls) * Destroy(hs=ls))),

        # Oscillator unitary group rules
        ('phase2', (
            pattern_head(pattern(Phase, u, hs=ls), pattern(Phase, v, hs=ls)),
            lambda ls, u, v: Phase.create(u + v, hs=ls))),
        ('displace2', (
            pattern_head(
                pattern(Displace, u, hs=ls),
                pattern(Displace, v, hs=ls)),
            lambda ls, u, v: (
                exp((u * v.conjugate() - u.conjugate() * v) / 2) *
                Displace.create(u + v, hs=ls)))),

        ('dphase', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Phase, u, hs=ls)),
            lambda ls, u:
                exp(I * u) * Phase.create(u, hs=ls) * Destroy(hs=ls))),
        ('ddisplace', (
            pattern_head(pattern(Destroy, hs=ls), pattern(Displace, u, hs=ls)),
            lambda ls, u: Displace.create(u, hs=ls) * (Destroy(hs=ls) + u))),

        ('phasec', (
            pattern_head(pattern(Phase, u, hs=ls), pattern(Create, hs=ls)),
            lambda ls, u:
                exp(I * u) * Create(hs=ls) * Phase.create(u, hs=ls))),
        ('displacec', (
            pattern_head(pattern(Displace, u, hs=ls), pattern(Create, hs=ls)),
            lambda ls, u: (((Create(hs=ls) - u.conjugate()) *
                            Displace.create(u, hs=ls))))),

        ('phasesig', (
            pattern_head(pattern(Phase, u, hs=ls), localsigma),
            lambda ls, u, localsigma:
            exp(I * u * localsigma.index_j) * localsigma)),
        ('sigphase', (
            pattern_head(localsigma, pattern(Phase, u, hs=ls)),
            lambda ls, u, localsigma:
            exp(I * u * localsigma.index_k) * localsigma)),

        # Spin rules
        ('spin1', (
            pattern_head(pattern(Jplus, hs=ls), localsigma),
            lambda ls, localsigma:
                Jpjmcoeff(ls, localsigma.index_j, shift=True) *
                localsigma.raise_jk(j_incr=1))),

        ('spin2', (
            pattern_head(pattern(Jminus, hs=ls), localsigma),
            lambda ls, localsigma:
                Jmjmcoeff(ls, localsigma.index_j, shift=True) *
                localsigma.raise_jk(j_incr=-1))),

        ('spin3', (
            pattern_head(pattern(Jz, hs=ls), localsigma),
            lambda ls, localsigma:
                Jzjmcoeff(ls, localsigma.index_j, shift=True) * localsigma)),

        ('spin4', (
            pattern_head(localsigma, pattern(Jplus, hs=ls)),
            lambda ls, localsigma:
                Jmjmcoeff(ls, localsigma.index_k, shift=True) *
                localsigma.raise_jk(k_incr=-1))),

        ('spin5', (
            pattern_head(localsigma, pattern(Jminus, hs=ls)),
            lambda ls, localsigma:
                Jpjmcoeff(ls, localsigma.index_k, shift=True) *
                localsigma.raise_jk(k_incr=+1))),

        ('spin6', (
            pattern_head(localsigma, pattern(Jz, hs=ls)),
            lambda ls, localsigma:
                Jzjmcoeff(ls, localsigma.index_k, shift=True) * localsigma)),

        # Normal ordering for angular momentum <=> all J_+ to the left, J_z to
        # center and J_- to the right
        ('spinord1', (
            pattern_head(pattern(Jminus, hs=ls), pattern(Jplus, hs=ls)),
            lambda ls: -2 * Jz(hs=ls) + Jplus(hs=ls) * Jminus(hs=ls))),

        ('spinord2', (
            pattern_head(pattern(Jminus, hs=ls), pattern(Jz, hs=ls)),
            lambda ls: Jz(hs=ls) * Jminus(hs=ls) + Jminus(hs=ls))),

        ('spinord3', (
            pattern_head(pattern(Jz, hs=ls), pattern(Jplus, hs=ls)),
            lambda ls: Jplus(hs=ls) * Jz(hs=ls) + Jplus(hs=ls))),
    ]))

    Adjoint._rules.update(check_rules_dict([
        ('uA', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u.conjugate() * A.adjoint())),
        ('plus', (
            pattern_head(A_plus),
            lambda A: OperatorPlus.create(*[o.adjoint()
                                            for o in A.operands]))),
        ('times', (
            pattern_head(A_times),
            lambda A: OperatorTimes.create(*[o.adjoint()
                                             for o in A.operands[::-1]]))),
        ('adjoint', (
            pattern_head(pattern(Adjoint, A)),
            lambda A: A)),
        ('create', (
            pattern_head(pattern(Create, hs=ls)),
            lambda ls: Destroy(hs=ls))),
        ('destroy', (
            pattern_head(pattern(Destroy, hs=ls)),
            lambda ls: Create(hs=ls))),
        ('jplus', (
            pattern_head(pattern(Jplus, hs=ls)),
            lambda ls: Jminus(hs=ls))),
        ('jminus', (
            pattern_head(pattern(Jminus, hs=ls)),
            lambda ls: Jplus(hs=ls))),
        ('jz', (
            pattern_head(pattern(Jz, hs=ls)),
            lambda ls: Jz(hs=ls))),
        ('sig', (
            pattern_head(pattern(LocalSigma, ra, rb, hs=ls)),
            lambda ls, ra, rb: LocalSigma(rb, ra, hs=ls))),
        ('proj', (
            pattern_head(pattern(LocalProjector, ra, hs=ls)),
            lambda ls, ra: LocalProjector(ra, hs=ls))),
    ]))

    Displace._rules.update(check_rules_dict([
        ('zero', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    Phase._rules.update(check_rules_dict([
        ('zero', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    Squeeze._rules.update(check_rules_dict([
        ('zero', (
            pattern_head(0, hs=ls), lambda ls: IdentityOperator))
    ]))
    LocalSigma._rules.update(check_rules_dict([
        ('projector', (
            pattern_head(n, n, hs=ls), lambda n, ls: LocalProjector(n, hs=ls)))
    ]))

    OperatorTrace._rules.update(check_rules_dict([
        ('triv', (
            pattern_head(A, over_space=TrivialSpace),
            lambda A: A)),
        ('zero', (
            pattern_head(ZeroOperator, over_space=h1),
            lambda h1: ZeroOperator)),
        ('id', (
            pattern_head(IdentityOperator, over_space=h1),
            lambda h1: h1.dimension * IdentityOperator)),
        ('plus', (
            pattern_head(A_plus, over_space=h1),
            lambda h1, A: OperatorPlus.create(
                *[OperatorTrace.create(o, over_space=h1)
                  for o in A.operands]))),
        ('adjoint', (
            pattern_head(pattern(Adjoint, A), over_space=h1),
            lambda h1, A: Adjoint.create(
                OperatorTrace.create(A, over_space=h1)))),
        ('uA', (
            pattern_head(pattern(ScalarTimesOperator, u, A), over_space=h1),
            lambda h1, u, A: u * OperatorTrace.create(A, over_space=h1))),
        ('prodspace', (
            pattern_head(A, over_space=H_ProductSpace),
            lambda H, A: decompose_space(H, A))),
        ('create', (
            pattern_head(pattern(Create, hs=ls), over_space=ls),
            lambda ls: ZeroOperator)),
        ('destroy', (
            pattern_head(pattern(Destroy, hs=ls), over_space=ls),
            lambda ls: ZeroOperator)),
        ('sigma', (
            pattern_head(pattern(LocalSigma, n, m, hs=ls), over_space=ls),
            lambda ls, n, m: KroneckerDelta(n, m) * IdentityOperator)),
        ('proj', (
            pattern_head(pattern(LocalProjector, n, hs=ls), over_space=ls),
            lambda ls, n: IdentityOperator)),
        ('factor', (
            pattern_head(A, over_space=ls),
            lambda ls, A: factor_for_trace(ls, A))),
    ]))

    PseudoInverse._rules.update(check_rules_dict([
        ('sig', (
            pattern_head(pattern(LocalSigma, m, n, hs=ls)),
            lambda ls, m, n: LocalSigma(n, m, hs=ls))),
        ('proj', (
            pattern_head(pattern(LocalProjector, m, hs=ls)),
            lambda ls, m: LocalSigma(m, hs=ls))),
    ]))

    Commutator._rules.update(check_rules_dict([
        ('AA', (
            pattern_head(A, A), lambda A: ZeroOperator)),
        ('uAvB', (
            pattern_head(
                pattern(ScalarTimesOperator, u, A),
                pattern(ScalarTimesOperator, v, B)),
            lambda u, v, A, B: u * v * Commutator.create(A, B))),
        ('vAB', (
            pattern_head(pattern(ScalarTimesOperator, v, A), B),
            lambda v, A, B: v * Commutator.create(A, B))),
        ('AvB', (
            pattern_head(A, pattern(ScalarTimesOperator, v, B)),
            lambda v, A, B: v * Commutator.create(A, B))),

        # special known commutators
        ('crdest', (
            pattern_head(pattern(Create, hs=ls), pattern(Destroy, hs=ls)),
            lambda ls: ScalarTimesOperator(-1, IdentityOperator))),
        # the remaining  rules basically defer to OperatorTimes; just writing
        # out the commutator will generate something simple
        ('expand1', (
            pattern_head(
                wc('A', head=(
                    Create, Destroy, LocalSigma, LocalProjector, Phase,
                    Displace)),
                wc('B', head=(
                    Create, Destroy, LocalSigma, LocalProjector, Phase,
                    Displace))),
            lambda A, B: A * B - B * A)),
        ('expand2', (
            pattern_head(
                wc('A', head=(LocalSigma, LocalProjector, Jplus, Jminus, Jz)),
                wc('B', head=(LocalSigma, LocalProjector, Jplus, Jminus, Jz))),
            lambda A, B: A * B - B * A)),
    ]))

    def pull_constfactor_from_sum(u, A, indranges):
        bound_symbols = set([r.index_symbol for r in indranges])
        if len(all_symbols(u).intersection(bound_symbols)) == 0:
            return u * OperatorIndexedSum.create(A, *indranges)
        else:
            raise CannotSimplify()

    OperatorIndexedSum._rules.update(check_rules_dict([
        ('R001', (  # sum over zero -> zero
            pattern_head(ZeroOperator, indranges__),
            lambda indranges: ZeroOperator)),
        ('R002', (  # pull constant prefactor out of sum
            pattern_head(pattern(ScalarTimesOperator, u, A), indranges__),
            lambda u, A, indranges:
                pull_constfactor_from_sum(u, A, indranges))),
    ]))


_algebraic_rules()
