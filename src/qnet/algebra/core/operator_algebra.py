r"""
This module features classes and functions to define and manipulate symbolic
Operator expressions.  For more details see :ref:`operator_algebra`.

For a list of all properties and methods of an operator object, see the
documentation for the basic :class:`Operator` class.
"""
import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from itertools import product as cartesian_product

from sympy import sympify

from qnet.utils.properties_for_args import properties_for_args
from .abstract_quantum_algebra import (
    ScalarTimesQuantumExpression, QuantumExpression, QuantumSymbol,
    QuantumOperation, QuantumPlus, QuantumTimes, SingleQuantumOperation,
    QuantumAdjoint, QuantumIndexedSum, QuantumDerivative, ensure_local_space)
from .algebraic_properties import (
    assoc, assoc_indexed, commutator_order, delegate_to_method,
    disjunct_hs_zero, filter_neutral, implied_local_space, match_replace,
    match_replace_binary, orderby, scalars_to_op, indexed_sum_over_const,
    indexed_sum_over_kronecker, collect_summands)
from .exceptions import CannotSimplify
from .hilbert_space_algebra import (
    HilbertSpace, LocalSpace, ProductSpace, TrivialSpace, )
from .scalar_algebra import Scalar, ScalarValue, is_scalar
from ..pattern_matching import pattern, pattern_head, wc
from ...utils.indices import (
    SymbolicLabelBase, IdxSym, IndexOverFockSpace, FockIndex)
from ...utils.ordering import FullCommutativeHSOrder
from ...utils.singleton import Singleton, singleton_object

sympyOne = sympify(1)

# for hilbert space dimensions less than or equal to this,
# compute numerically PseudoInverse and NullSpaceProjector representations
DENSE_DIMENSION_LIMIT = 1000

__all__ = [
    'Adjoint', 'LocalOperator', 'LocalSigma', 'NullSpaceProjector', 'Operator',
    'OperatorPlus', 'OperatorPlusMinusCC', 'OperatorSymbol', 'OperatorTimes',
    'OperatorTrace', 'PseudoInverse', 'ScalarTimesOperator',
    'LocalProjector', 'adjoint', 'rewrite_with_operator_pm_cc',
    'decompose_space', 'factor_coeff', 'factor_for_trace', 'get_coeffs', 'II',
    'IdentityOperator', 'ZeroOperator', 'OperatorDerivative', 'Commutator',
    'OperatorIndexedSum', 'tr']

__private__ = []
# anything not in __all__ must be in __private__


###############################################################################
# Abstract base classes
###############################################################################


class Operator(QuantumExpression, metaclass=ABCMeta):
    """Base class for all quantum operators."""

    def pseudo_inverse(self):
        """Pseudo-inverse $\Op{X}^+$ of the operator $\Op{X}$

        It is defined via the relationship

        .. math::

            \Op{X} \Op{X}^+ \Op{X} =  \Op{X}  \\
            \Op{X}^+ \Op{X} \Op{X}^+ =  \Op{X}^+  \\
            (\Op{X}^+ \Op{X})^\dagger = \Op{X}^+ \Op{X}  \\
            (\Op{X} \Op{X}^+)^\dagger = \Op{X} \Op{X}^+
        """
        return self._pseudo_inverse()

    @abstractmethod
    def _pseudo_inverse(self):
        raise NotImplementedError(self.__class__.__name__)

    def expand_in_basis(self, basis_states=None, hermitian=False):
        """Write the operator as an expansion into all
        :class:`KetBras <.KetBra>`
        spanned by `basis_states`.

        Args:
            basis_states (list or None): List of basis states (:class:`.State`
                instances) into which to expand the operator. If None, use the
                operator's `space.basis_states`
            hermitian (bool): If True, assume that the operator is Hermitian
                and represent all elements in the lower triangle of the
                expansion via :class:`OperatorPlusMinusCC`. This is meant to
                enhance readability

        Raises:
            .BasisNotSetError: If `basis_states` is None and the operator's
                Hilbert space has no well-defined basis

        Example:

            >>> hs = LocalSpace(1, basis=('g', 'e'))
            >>> op = LocalSigma('g', 'e', hs=hs) + LocalSigma('e', 'g', hs=hs)
            >>> print(ascii(op, sig_as_ketbra=False))
            sigma_e,g^(1) + sigma_g,e^(1)
            >>> print(ascii(op.expand_in_basis()))
            |e><g|^(1) + |g><e|^(1)
            >>> print(ascii(op.expand_in_basis(hermitian=True)))
            |g><e|^(1) + c.c.
        """
        from qnet.algebra.core.state_algebra import KetBra
        # KetBra is imported locally to avoid circular imports
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
                op_ij = (ket_i.dag() * self * ket_j).expand()
                ketbra = KetBra(ket_i, ket_j)
                term = op_ij * ketbra
                if term is not ZeroOperator:
                    if i == j:
                        diag_terms.append(op_ij * ketbra)
                    else:
                        terms.append(op_ij * ketbra)
        if hermitian:
            res = OperatorPlus.create(*diag_terms)
            if len(terms) > 0:
                res = res + OperatorPlusMinusCC(OperatorPlus.create(*terms))
            return res
        else:
            return (
                OperatorPlus.create(*diag_terms) +
                OperatorPlus.create(*terms))


class LocalOperator(Operator, metaclass=ABCMeta):
    """Base class for "known" operators on a :class:`LocalSpace`

    All :class:`LocalOperator` instances have known algebraic properties and a
    fixed associated identifier (symbol) that is used when printing that
    operator. A custom identifier can be used through the associated
    :class:`.LocalSpace`'s `local_identifiers` parameter. For example::

        >>> hs1_custom = LocalSpace(1, local_identifiers={'Destroy': 'b'})
        >>> b = Destroy(hs=hs1_custom)
        >>> ascii(b)
        'b^(1)'

    Note:
        It is recommended that subclasses use the :func:`.properties_for_args`
        class decorator if they define any position arguments (via the
        :attr:`_arg_names` class attribute)
    """

    simplifications = [implied_local_space(keys=['hs', ]), ]

    _identifier = None  # must be overridden by subclasses!
    _dagger = False  # do representations include a dagger?
    _arg_names = ()  # names of args that can be passed to __init__
    _scalar_args = True  # convert args to Scalar?
    _hs_cls = LocalSpace  # allowed type of `hs`
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9().+-]+)?$')

    def __init__(self, *args, hs):
        if len(args) != len(self._arg_names):
            raise ValueError("expected %d arguments, gotten %d"
                             % (len(self._arg_names), len(args)))
        if self._scalar_args:
            args = [ScalarValue.create(arg) for arg in args]
        for i, arg_name in enumerate(self._arg_names):
            self.__dict__['_%s' % arg_name] = args[i]
        hs = ensure_local_space(hs, cls=self._hs_cls)
        self._hs = hs
        if self._identifier is None:
            raise TypeError(
                r"Can't instantiate abstract class %s with undefined "
                r"_identifier" % self.__class__.__name__)
        self._args = args
        super().__init__(*args, hs=hs)

    @property
    def space(self):
        """Hilbert space of the operator (:class:`.LocalSpace` instance)"""
        return self._hs

    @property
    def args(self):
        """The positional arguments used for instantiating the operator"""
        return tuple(self._args)

    @property
    def kwargs(self):
        """The keyword arguments used for instantiating the operator"""
        return OrderedDict([('hs', self._hs)])

    @property
    def identifier(self):
        """The identifier (symbol) that is used when printing the operator.

        A custom identifier can be used through the associated
        :class:`.LocalSpace`'s `local_identifiers` parameter. For example::

            >>> a = Destroy(hs=1)
            >>> a.identifier
            'a'
            >>> hs1_custom = LocalSpace(1, local_identifiers={'Destroy': 'b'})
            >>> b = Destroy(hs=hs1_custom)
            >>> b.identifier
            'b'
            >>> ascii(b)
            'b^(1)'
        """

        identifier = self._hs._local_identifiers.get(
            self.__class__.__name__, self._identifier)
        if not self._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'"
                % (identifier, self._rx_identifier.pattern))
        return identifier

    def _diff(self, sym):
        return OperatorDerivative(self, derivs={sym: 1})

    def _simplify_scalar(self, func):
        if self._scalar_args:
            args = [arg.simplify_scalar(func=func) for arg in self.args]
            return self.create(*args, hs=self.space)
        else:
            return super()._simplify_scalar(func=func)


###############################################################################
# Operator algebra elements
###############################################################################


class OperatorSymbol(QuantumSymbol, Operator):
    """Symbolic operator

    See :class:`.QuantumSymbol`.
    """
    def _pseudo_inverse(self):
        return PseudoInverse(self)


@singleton_object
class IdentityOperator(Operator, metaclass=Singleton):
    """``IdentityOperator`` constant (singleton) object."""

    _order_index = 2

    @property
    def space(self):
        """:class:`.TrivialSpace`"""
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def _diff(self, sym):
        return ZeroOperator

    def _adjoint(self):
        return self

    def _pseudo_inverse(self):
        return self


II = IdentityOperator


@singleton_object
class ZeroOperator(Operator, metaclass=Singleton):
    """``ZeroOperator`` constant (singleton) object."""

    _order_index = 2

    @property
    def space(self):
        """:class:`.TrivialSpace`"""
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def _diff(self, sym):
        return self

    def _adjoint(self):
        return self

    def _pseudo_inverse(self):
        return self


@properties_for_args
class LocalSigma(LocalOperator):
    r'''Level flip operator between two levels of a :class:`.LocalSpace`

    .. math::

        \Op{\sigma}_{jk}^{\rm hs} =
        \left| j\right\rangle_{\rm hs} \left \langle k \right |_{\rm hs}

    For $j=k$ this becomes a projector $\Op{P}_k$ onto the eigenstate
    $\ket{k}$; see :class:`LocalProjector`.

    Args:
        j (int or str): The label or index identifying $\ket{j}$
        k (int or str):  The label or index identifying $\ket{k}$
        hs (LocalSpace or int or str): The Hilbert space on which the
            operator acts. If an :class:`int` or a :class:`str`, an implicit
            Hilbert space will be constructed as a subclass of
            :class:`LocalSpace`, as configured by :func:`init_algebra`.

    Note:
        The parameters `j` or `k` may be an integer or a string. A string
        refers to the label of an eigenstate in the basis of `hs`, which needs
        to be set. An integer refers to the (zero-based) index of eigenstate of
        the Hilbert space. This works if `hs` has an unknown dimension.
        Assuming the Hilbert space has a defined basis, using integer or string
        labels is equivalent::

            >>> hs = LocalSpace('tls', basis=('g', 'e'))
            >>> LocalSigma(0, 1, hs=hs) == LocalSigma('g', 'e', hs=hs)
            True

    Raises:
        ValueError: If `j` or `k` are invalid value for the given `hs`

    Printers should represent this operator either in braket notation, or using
    the operator identifier

        >>> LocalSigma(0, 1, hs=0).identifier
        'sigma'

    For ``j == k``, an alternative (fixed) identifier may be used

        >>> LocalSigma(0, 0, hs=0)._identifier_projector
        'Pi'
    '''

    _identifier = "sigma"
    _identifier_projector = "Pi"
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*$')
    _arg_names = ('j', 'k')
    _scalar_args = False  # args are labels, not scalar coefficients
    _rules = OrderedDict()
    simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def __init__(self, j, k, *, hs):
        if isinstance(hs, (str, int)):
            hs = self._default_hs_cls(hs)
        hs._unpack_basis_label_or_index(j)  # for applying checks only ...
        hs._unpack_basis_label_or_index(k)  # ... (disregard returned tuple)
        if hs.has_basis:
            # normalize integer i/j to str label, if possible
            if isinstance(j, int):
                j = hs.basis_labels[j]
            if isinstance(k, int):
                k = hs.basis_labels[k]
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
            try:
                return self.space.basis_labels.index(self.j)
            except ValueError:
                raise ValueError(
                    "%r is not one of the basis labels %r"
                    % (self.j, self.space.basis_labels))

    @property
    def index_k(self):
        """Index `k` or (zero-based) index of the label `k` in the basis"""
        if isinstance(self.k, (int, SymbolicLabelBase)):
            return self.k
        else:
            try:
                return self.space.basis_labels.index(self.k)
            except ValueError:
                raise ValueError(
                    "%r is not one of the basis labels %r"
                    % (self.k, self.space.basis_labels))

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

    def _adjoint(self):
        return LocalSigma(j=self.k, k=self.j, hs=self.space)

    def _pseudo_inverse(self):
        return self._adjoint()


def LocalProjector(j, *, hs):
    """A projector onto a specific level of a :class:`.LocalSpace`

    Args:
        j (int or str): The label or index identifying the state onto which
            is projected
        hs (HilbertSpace): The Hilbert space on which the operator acts
    """
    return LocalSigma(j, j, hs=hs)


###############################################################################
# Algebra Operations
###############################################################################


class OperatorPlus(QuantumPlus, Operator):
    """Sum of Operators"""

    _neutral_element = ZeroOperator
    _binary_rules = OrderedDict()
    simplifications = [
        assoc, scalars_to_op, orderby, collect_summands, match_replace_binary]

    def _pseudo_inverse(self):
        return PseudoInverse(self)


class OperatorTimes(QuantumTimes, Operator):
    """Product of operators

    This serves both as a product within a Hilbert space as well as a tensor
    product."""

    _neutral_element = IdentityOperator
    _binary_rules = OrderedDict()
    simplifications = [assoc, orderby, filter_neutral, match_replace_binary]

    def _pseudo_inverse(self):
        return self.__class__.create(
                *[o._pseudo_inverse() for o in reversed(self.operands)])


class ScalarTimesOperator(Operator, ScalarTimesQuantumExpression):
    """Product of a :class:`.Scalar` coefficient and an :class:`Operator`"""

    _rules = OrderedDict()
    simplifications = [match_replace, ]

    @staticmethod
    def has_minus_prefactor(c):
        """
        For a scalar object c, determine whether it is prepended by a "-" sign.
        """
        # TODO: check if this is necessary; if yes, move
        cs = str(c).strip()
        return cs[0] == "-"

    def _pseudo_inverse(self):
        c, t = self.operands
        return t.pseudo_inverse() / c

    def __eq__(self, other):
        # TODO: review, and add this to ScalarTimesQuantumExpression
        if self.term is IdentityOperator and is_scalar(other):
            return self.coeff == other
        return super().__eq__(other)

    def __hash__(self):
        # TODO: review, and add this to ScalarTimesQuantumExpression
        return super().__hash__()

    def _adjoint(self):
        return ScalarTimesOperator(self.coeff.conjugate(), self.term.adjoint())


class OperatorDerivative(QuantumDerivative, Operator):
    """Symbolic partial derivative of an operator

    See :class:`.QuantumDerivative`.
    """
    def _pseudo_inverse(self):
        return PseudoInverse(self)


class Commutator(QuantumOperation, Operator):
    r'''Commutator of two operators

    .. math::

        [\Op{A}, \Op{B}] = \Op{A}\Op{B} - \Op{A}\Op{B}

    '''

    _rules = OrderedDict()
    simplifications = [
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

    def doit(self, classes=None, recursive=True, **kwargs):
        """Write out commutator

        Write out the commutator according to its definition
        $[\Op{A}, \Op{B}] = \Op{A}\Op{B} - \Op{A}\Op{B}$.

        See :meth:`.Expression.doit`.
        """
        return super().doit(classes, recursive, **kwargs)

    def _doit(self, **kwargs):
        return self.A * self.B - self.B * self.A

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

    def _adjoint(self):
        return Commutator(self.B.adjoint(), self.A.adjoint())

    def _pseudo_inverse(self):
        return PseudoInverse(self)


class OperatorTrace(SingleQuantumOperation, Operator):
    r'''(Partial) trace of an operator

    Trace of an operator `op` ($\Op{O}) over the degrees
    of freedom of a Hilbert space `over_space` ($\mathcal{H}$):

    .. math::

        {\rm Tr}_{\mathcal{H}} \Op{O}

    Args:
        over_space (.HilbertSpace): The degrees of freedom to trace over
        op (Operator): The operator to take the trace of.
    '''
    _rules = OrderedDict()
    simplifications = [
        scalars_to_op, implied_local_space(keys=['over_space', ]),
        match_replace, ]

    def __init__(self, op, *, over_space):
        if isinstance(over_space, (int, str)):
            over_space = self._default_hs_cls(over_space)
        assert isinstance(over_space, HilbertSpace)
        self._over_space = over_space
        super().__init__(op, over_space=over_space)
        self._space = None

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

    def _diff(self, sym):
        s = self._over_space
        o = self.operand
        return OperatorTrace.create(o._diff(sym), over_space=s)

    def _adjoint(self):
        # there is a rule Tr[A^\dagger] -> Tr[A]^\dagger, which we don't want
        # to counteract here with an inverse rule
        return Adjoint(self)

    def _pseudo_inverse(self):
        return PseudoInverse(self)


class Adjoint(QuantumAdjoint, Operator):
    """Symbolic Adjoint of an operator"""

    simplifications = [
        scalars_to_op, delegate_to_method('_adjoint')]
    # The reason that Adjoint does not have have `match_replace` in
    # `simplifications`, respectively a `_rules` class attribute is that the
    # `_adjoint` property that we delegate to is mandatory. Thus, if we had
    # rules on top of that, it would create the confusing situation of the rule
    # contradicting the `_adjoint` property.

    def _pseudo_inverse(self):
        return self.operand.pseudo_inverse().adjoint()


class OperatorPlusMinusCC(SingleQuantumOperation, Operator):
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

    def _diff(self, sym):
        return OperatorPlusMinusCC(
            self.operands._diff(sym), sign=self._sign)

    def _adjoint(self):
        return OperatorPlusMinusCC(self.operand.adjoint(), sign=self._sign)

    def _pseudo_inverse(self):
        return PseudoInverse(self.doit())

    def doit(self, classes=None, recursive=True, **kwargs):
        """Write out the complex conjugate summand

        See :meth:`.Expression.doit`.
        """
        return super().doit(classes, recursive, **kwargs)

    def _doit(self, **kwargs):
        if self._sign > 0:
            return self.operand + self.operand.adjoint()
        else:
            return self.operand - self.operand.adjoint()


class PseudoInverse(SingleQuantumOperation, Operator):
    r"""Unevaluated pseudo-inverse $\Op{X}^+$ of an operator $\Op{X}$

    It is defined via the relationship

    .. math::

        \Op{X} \Op{X}^+ \Op{X} =  \Op{X}  \\
        \Op{X}^+ \Op{X} \Op{X}^+ =  \Op{X}^+  \\
        (\Op{X}^+ \Op{X})^\dagger = \Op{X}^+ \Op{X}  \\
        (\Op{X} \Op{X}^+)^\dagger = \Op{X} \Op{X}^+
    """
    simplifications = [
        scalars_to_op, delegate_to_method('_pseudo_inverse')]
    # `PseudoInverse` does not use rules because it delegates to
    # `_pseudo_inverse`, cf. `Adjoint`

    def _expand(self):
        return self

    def _pseudo_inverse(self):
        return self.operand

    def _adjoint(self):
        return Adjoint(self)


class NullSpaceProjector(SingleQuantumOperation, Operator):
    r"""Projection operator onto the nullspace of its operand

    Returns the operator :math:`\mathcal{P}_{{\rm Ker} X}` with

    .. math::

        X \mathcal{P}_{{\rm Ker} X}
          = 0
        \Leftrightarrow
        X (1 - \mathcal{P}_{{\rm Ker} X})
          = X \\
        \mathcal{P}_{{\rm Ker} X}^\dagger
          = \mathcal{P}_{{\rm Ker} X}
          = \mathcal{P}_{{\rm Ker} X}^2
    """

    _rules = OrderedDict()
    simplifications = [scalars_to_op, match_replace, ]

    def _expand(self):
        return self

    def _adjoint(self):
        return Adjoint(self)

    def _pseudo_inverse(self):
        return PseudoInverse(self)


class OperatorIndexedSum(QuantumIndexedSum, Operator):
    """Indexed sum over operators"""

    _rules = OrderedDict()
    simplifications = [
        assoc_indexed, scalars_to_op, indexed_sum_over_kronecker,
        indexed_sum_over_const, match_replace, ]

    def _pseudo_inverse(self):
        return PseudoInverse(self)


###############################################################################
# Constructor Routines
###############################################################################


tr = OperatorTrace.create


###############################################################################
# Auxilliary routines
###############################################################################


def factor_for_trace(ls: HilbertSpace, op: Operator) -> Operator:
    r'''Given a :class:`.LocalSpace` `ls` to take the partial trace over and an
    operator `op`, factor the trace such that operators acting on disjoint
    degrees of freedom are pulled out of the trace. If the operator acts
    trivially on ls the trace yields only a pre-factor equal to the dimension
    of ls. If there are :class:`LocalSigma` operators among a product, the
    trace's cyclical property is used to move to sandwich the full product by
    :class:`LocalSigma` operators:

    .. math::

        {\rm Tr} A \sigma_{jk} B = {\rm Tr} \sigma_{jk} B A \sigma_{jj}

    Args:
        ls: Degree of Freedom to trace over
        op: Operator to take the trace of

    Returns:
        The (partial) trace over the operator's spc-degrees of freedom
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
            for j, r in enumerate(rest):
                if isinstance(r, LocalSigma):
                    m = r.j
                    rest = (
                        rest[j:] + rest[:j] +
                        [LocalSigma.create(m, m, hs=ls), ])
                    break
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

    Args:
        H (ProductSpace): The full space.
        A (Operator):

    Returns:
        Operator: Iterative partial trace expression
    """
    return OperatorTrace.create(
        OperatorTrace.create(A, over_space=H.operands[-1]),
        over_space=ProductSpace.create(*H.operands[:-1]))


def get_coeffs(expr, expand=False, epsilon=0.):
    """Create a dictionary with all Operator terms of the expression
    (understood as a sum) as keys and their coefficients as values.

    The returned object is a defaultdict that return 0. if a term/key
    doesn't exist.

    Args:
        expr: The operator expression to get all coefficients from.
        expand: Whether to expand the expression distributively.
        epsilon: If non-zero, drop all Operators with coefficients that have
            absolute value less than epsilon.

    Returns:
        dict: A dictionary ``{op1: coeff1, op2: coeff2, ...}``
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


def _coeff_term(op):
    # TODO: remove
    if isinstance(op, ScalarTimesOperator):
        return op.coeff, op.term
    elif is_scalar(op):
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


###############################################################################
# Extra ("manual") simplifications
###############################################################################


def rewrite_with_operator_pm_cc(expr):
    """Try to rewrite expr using :class:`OperatorPlusMinusCC`

    Example:

        >>> A = OperatorSymbol('A', hs=1)
        >>> sum = A + A.dag()
        >>> sum2 = rewrite_with_operator_pm_cc(sum)
        >>> print(ascii(sum2))
        A^(1) + c.c.
    """
    # TODO: move this to the toolbox
    from qnet.algebra.toolbox.core import temporary_rules

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

    A = wc("A", head=Operator)
    B = wc("B", head=Operator)
    c = wc("c", head=Scalar)
    d = wc("d", head=Scalar)

    with temporary_rules(OperatorPlus, clear=True):
        OperatorPlus.add_rule(
            'PM1', pattern_head(A, B), _combine_operator_p_cc)
        OperatorPlus.add_rule(
            'PM2',
            pattern_head(pattern(ScalarTimesOperator, -1, B), A),
            _combine_operator_m_cc)
        OperatorPlus.add_rule(
            'PM3',
            pattern_head(
                pattern(ScalarTimesOperator, c, A),
                pattern(ScalarTimesOperator, d, B)),
            _scal_combine_operator_pm_cc)
        return expr.rebuild()


Operator._zero = ZeroOperator
Operator._one = IdentityOperator
Operator._base_cls = Operator
Operator._scalar_times_expr_cls = ScalarTimesOperator
Operator._plus_cls = OperatorPlus
Operator._times_cls = OperatorTimes
Operator._adjoint_cls = Adjoint
Operator._indexed_sum_cls = OperatorIndexedSum
Operator._derivative_cls = OperatorDerivative
