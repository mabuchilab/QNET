"""
The specification of a quantum mechanics symbolic super-operator algebra.
See :ref:`super_operator_algebra` for more details.
"""
from abc import ABCMeta
from collections import OrderedDict, defaultdict

from numpy import (array as np_array, sqrt as np_sqrt)
from numpy.linalg import eigh
from sympy import (I, Matrix as SympyMatrix, sqrt)

from .abstract_algebra import Operation
from .abstract_quantum_algebra import (
    ScalarTimesQuantumExpression, QuantumExpression, QuantumSymbol,
    QuantumOperation, QuantumPlus, QuantumTimes, QuantumAdjoint,
    QuantumDerivative)
from .algebraic_properties import (
    assoc, filter_neutral, match_replace, match_replace_binary, orderby,
    delegate_to_method, collect_summands)
from .exceptions import BadLiouvillianError, CannotSymbolicallyDiagonalize
from .hilbert_space_algebra import TrivialSpace
from .matrix_algebra import Matrix
from .operator_algebra import (
    Operator, OperatorPlus, ZeroOperator, sympyOne, Adjoint, PseudoInverse)
from .scalar_algebra import is_scalar, One
from ...utils.ordering import DisjunctCommutativeHSOrder, KeyTuple
from ...utils.singleton import Singleton, singleton_object

__all__ = [
    'SPost', 'SPre', 'ScalarTimesSuperOperator', 'SuperAdjoint',
    'SuperOperator', 'SuperOperatorPlus', 'SuperOperatorSymbol',
    'SuperOperatorTimes', 'SuperOperatorTimesOperator', 'anti_commutator',
    'commutator', 'lindblad', 'liouvillian', 'liouvillian_normal_form',
    'IdentitySuperOperator', 'ZeroSuperOperator', 'SuperOperatorDerivative']

__private__ = ['SuperCommutativeHSOrder']
# anything not in __all__ must be in __private__


###############################################################################
# Abstract base classes
###############################################################################


class SuperOperator(QuantumExpression, metaclass=ABCMeta):
    """Base class for super-operators"""
    def __mul__(self, other):
        if isinstance(other, Operator):
            return SuperOperatorTimesOperator.create(self, other)
        else:
            return super().__mul__(other)


###############################################################################
# Superoperator algebra elements
###############################################################################


class SuperOperatorSymbol(QuantumSymbol, SuperOperator):
    """Symbolic super-operator

    See :class:`.QuantumSymbol`.
    """
    pass


@singleton_object
class IdentitySuperOperator(SuperOperator, metaclass=Singleton):
    """Neutral element for product of super-operators"""

    _order_index = 2

    @property
    def space(self):
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def _diff(self, sym):
        return ZeroSuperOperator

    def _adjoint(self):
        return self

    def _expand(self):
        return self


@singleton_object
class ZeroSuperOperator(SuperOperator, metaclass=Singleton):
    """Neutral element for sum of super-operators"""

    _order_index = 2

    @property
    def space(self):
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def _diff(self, sym):
        return self

    def _adjoint(self):
        return self

    def _expand(self):
        return self


###############################################################################
# Algebra Operations
###############################################################################


class SuperOperatorPlus(QuantumPlus, SuperOperator):
    """A sum of super-operators"""

    _neutral_element = ZeroSuperOperator
    _binary_rules = OrderedDict()
    simplifications = [assoc, orderby, collect_summands, match_replace_binary]


class SuperCommutativeHSOrder(DisjunctCommutativeHSOrder):
    """Ordering class that acts like DisjunctCommutativeHSOrder, but also
    commutes any `SPost` and `SPre`"""
    def __lt__(self, other):
        if isinstance(self.op, SPre) and isinstance(other.op, SPost):
            return True
        elif isinstance(self.op, SPost) and isinstance(other.op, SPre):
            return False
        else:
            return DisjunctCommutativeHSOrder.__lt__(self, other)


class SuperOperatorTimes(QuantumTimes, SuperOperator):
    """Product of super-operators"""

    _neutral_element = IdentitySuperOperator
    _binary_rules = OrderedDict()  # see end of module
    simplifications = [assoc, orderby, filter_neutral, match_replace_binary]

    order_key = SuperCommutativeHSOrder

    @classmethod
    def create(cls, *ops):
        # TODO: Add this functionality to QuantumTimes
        if any(o == ZeroSuperOperator for o in ops):
            return ZeroSuperOperator
        return super().create(*ops)


class ScalarTimesSuperOperator(SuperOperator, ScalarTimesQuantumExpression):
    """Product of a :class:`.Scalar` coefficient and a
    :class:`SuperOperator`"""

    def _adjoint(self):
        pass

    _rules = OrderedDict()  # see end of module
    simplifications = [match_replace, ]

#    def _pseudo_inverse(self):
#        c, t = self.operands
#        return t.pseudo_inverse() / c


class SuperAdjoint(QuantumAdjoint, SuperOperator):
    r"""Adjoint of a super-operator

    The mathematical notation for this is typically

    .. math::
        {\rm SuperAdjoint}(\mathcal{L}) =: \mathcal{L}^*

    and for any super operator :math:`\mathcal{L}`, its super-adjoint
    :math:`\mathcal{L}^*` satisfies for any pair of operators :math:`M,N`:

    .. math::
        {\rm Tr}[M (\mathcal{L}N)] = Tr[(\mathcal{L}^*M)  N]

    """

    simplifications = [delegate_to_method('_adjoint')]

    def __init__(self, operand):
        super().__init__(operand)


class SPre(SuperOperator, Operation):
    """Linear pre-multiplication operator

    Acting ``SPre(A)`` on an operator ``B`` just yields the product ``A * B``
    """

    _rules = OrderedDict()  # see end of module
    simplifications = [match_replace, ]

    _order_name = 'A_SPre'  # "SPre" should go before "SPost"

    @property
    def space(self):
        return self.operands[0].space

    def _expand(self):
        oe = self.operands[0].expand()
        if isinstance(oe, OperatorPlus):
            return sum(SPre.create(oet) for oet in oe.operands)
        return SPre.create(oe)

    def _simplify_scalar(self, func):
        return self.create(self.operands[0].simplify_scalar(func=func))

    def _diff(self, sym):
        return SuperOperatorDerivative(self, {sym: 1})

    def _adjoint(self):
        return SPost(self.operands[0])


class SPost(SuperOperator, Operation):
    """Linear post-multiplication operator

        Acting ``SPost(A)`` on an operator ``B`` just yields the reversed
        product ``B * A``.
    """

    _order_index = -1

    _rules = OrderedDict()  # see end of module
    simplifications = [match_replace, ]

    _order_name = 'B_SPost'  # "SPost" should go before "SPre"

    @property
    def space(self):
        return self.operands[0].space

    def _expand(self):
        oe = self.operands[0].expand()
        if isinstance(oe, OperatorPlus):
            return sum(SPost.create(oet) for oet in oe.operands)
        return SPost.create(oe)

    def _simplify_scalar(self, func):
        return self.create(self.operands[0].simplify_scalar(func=func))

    def _diff(self, sym):
        return SuperOperatorDerivative(self, {sym: 1})

    def _adjoint(self):
        return SPre(self.operands[0])


class SuperOperatorTimesOperator(Operator, Operation):
    """Application of a super-operator to an operator

    The result of this operation is(result is an :class:`Operator`
    """

    _rules = OrderedDict()  # see end of module
    simplifications = [match_replace, ]

    def __init__(self, sop, op):
        assert isinstance(sop, SuperOperator)
        assert isinstance(op, Operator)
        super().__init__(sop, op)

    @property
    def space(self):
        return self.sop.space * self.op.space

    @property
    def sop(self):
        return self.operands[0]

    @property
    def op(self):
        return self.operands[1]

    def _expand(self):
        sop, op = self.operands
        sope, ope = sop.expand(), op.expand()
        if isinstance(sope, SuperOperatorPlus):
            sopet = sope.operands
        else:
            sopet = (sope, )
        if isinstance(ope, OperatorPlus):
            opet = ope.operands
        else:
            opet = (ope, )

        return sum(st * ot for st in sopet for ot in opet)

    def _series_expand(self, param, about, order):
        sop, op = self.sop, self.op
        ope = op.series_expand(param, about, order)
        return tuple(sop * opet for opet in ope)

    def _simplify_scalar(self, func):
        sop, op = self.sop, self.op
        return sop.simplify_scalar(func=func) * op.simplify_scalar(func=func)

    def _adjoint(self):
        return Adjoint(self)

    def _pseudo_inverse(self):
        return PseudoInverse(self)

    def _diff(self, sym):
        return self.sop.diff(sym) * self.op + self.sop * self.op.diff(sym)


class SuperOperatorDerivative(QuantumDerivative, SuperOperator):
    """Symbolic partial derivative of a super-operator

    See :class:`.QuantumDerivative`.
    """
    pass


###############################################################################
# Constructor Routines
###############################################################################


def commutator(A, B=None):
    """Commutator of `A` and `B`

    If ``B != None``, return the commutator :math:`[A,B]`, otherwise return
    the super-operator :math:`[A,\cdot]`.  The super-operator :math:`[A,\cdot]`
    maps any other operator ``B`` to the commutator :math:`[A, B] = A B - B A`.

    Args:
        A: The first operator to form the commutator of.
        B: The second operator to form the commutator of, or None.

    Returns:
        SuperOperator: The linear superoperator :math:`[A,\cdot]`

    """
    if B:
        return A * B - B * A
    return SPre(A) - SPost(A)


def anti_commutator(A, B=None):
    """If ``B != None``, return the anti-commutator :math:`\{A,B\}`, otherwise
    return the super-operator :math:`\{A,\cdot\}`.  The super-operator
    :math:`\{A,\cdot\}` maps any other operator ``B`` to the anti-commutator
    :math:`\{A, B\} = A B + B A`.

    Args:
        A: The first operator to form all anti-commutators of.
        B: The second operator to form the anti-commutator of, or None.

    Returns:
        SuperOperator: The linear superoperator :math:`[A,\cdot]`

    """
    if B:
        return A * B + B * A
    return SPre(A) + SPost(A)


def lindblad(C):
    """Return the super-operator Lindblad term of the Lindblad operator `C`

    Return ``SPre(C) * SPost(C.adjoint()) - (1/2) *
    santi_commutator(C.adjoint()*C)``.  These are the super-operators
    :math:`\mathcal{D}[C]` that form the collapse terms of a Master-Equation.
    Applied to an operator :math:`X` they yield

    .. math::
        \mathcal{D}[C] X
            = C X C^\dagger - {1\over 2} (C^\dagger C X + X C^\dagger C)

    Args:
        C (Operator): The associated collapse operator

    Returns:
        SuperOperator: The Lindblad collapse generator.

    """
    if is_scalar(C):
        return ZeroSuperOperator
    return (
        SPre(C) * SPost(C.adjoint()) -
        (sympyOne/2) * anti_commutator(C.adjoint() * C))


def liouvillian(H, Ls=None):
    r"""Return the Liouvillian super-operator associated with `H` and `Ls`

    The Liouvillian :math:`\mathcal{L}` generates the Markovian-dynamics of a
    system via the Master equation:

    .. math::
        \dot{\rho} = \mathcal{L}\rho
            = -i[H,\rho] + \sum_{j=1}^n \mathcal{D}[L_j] \rho

    Args:
        H (Operator): The associated Hamilton operator
        Ls (sequence or Matrix): A sequence of Lindblad operators.

    Returns:
        SuperOperator: The Liouvillian super-operator.
    """
    if Ls is None:
        Ls = []
    elif isinstance(Ls, Matrix):
        Ls = Ls.matrix.ravel().tolist()
    summands = [-I * commutator(H), ]
    summands.extend([lindblad(L) for L in Ls])
    return SuperOperatorPlus.create(*summands)


###############################################################################
# Auxilliary routines
###############################################################################


def liouvillian_normal_form(L, symbolic = False):
    r"""Return a Hamilton operator ``H`` and a minimal list of collapse
    operators ``Ls`` that generate the liouvillian ``L``.

    A Liouvillian defined by a hermitian Hamilton operator :math:`H` and a
    vector of collapse operators
    :math:`\mathbf{L} = (L_1, L_2, \dots L_n)^T` is invariant under the
    following two operations:

    .. math::
        \left(H, \mathbf{L}\right) & \mapsto \left(H + {1\over 2i}\left(\mathbf{w}^\dagger \mathbf{L} - \mathbf{L}^\dagger \mathbf{w}\right), \mathbf{L} + \mathbf{w} \right) \\
        \left(H, \mathbf{L}\right) & \mapsto \left(H, \mathbf{U}\mathbf{L}\right)\\

    where :math:`\mathbf{w}` is just a vector of complex numbers and
    :math:`\mathbf{U}` is a complex unitary matrix.  It turns out that for
    quantum optical circuit models the set of collapse operators is often
    linearly dependent.  This routine tries to find a representation of the
    Liouvillian in terms of a Hamilton operator ``H`` with as few non-zero
    collapse operators ``Ls`` as possible.  Consider the following example,
    which results from a two-port linear cavity with a coherent input into the
    first port:

    >>> kappa_1, kappa_2 = sympy.symbols('kappa_1, kappa_2', positive = True)
    >>> Delta = sympy.symbols('Delta', real = True)
    >>> alpha = sympy.symbols('alpha')
    >>> H = (Delta * Create(hs=1) * Destroy(hs=1) +
    ...      (sqrt(kappa_1) / (2 * I)) *
    ...      (alpha * Create(hs=1) - alpha.conjugate() * Destroy(hs=1)))
    >>> Ls = [sqrt(kappa_1) * Destroy(hs=1) + alpha,
    ...       sqrt(kappa_2) * Destroy(hs=1)]
    >>> LL = liouvillian(H, Ls)
    >>> Hnf, Lsnf = liouvillian_normal_form(LL)
    >>> print(ascii(Hnf))
    -I*alpha*sqrt(kappa_1) * a^(1)H + I*sqrt(kappa_1)*conjugate(alpha) * a^(1) + Delta * a^(1)H * a^(1)
    >>> len(Lsnf)
    1
    >>> print(ascii(Lsnf[0]))
    sqrt(kappa_1 + kappa_2) * a^(1)

    In terms of the ensemble dynamics this final system is equivalent.
    Note that this function will only work for proper Liouvillians.


    Args:
        L (SuperOperator): The Liouvillian

    Returns:
        tuple: ``(H, Ls)``

    Raises:
        .BadLiouvillianError
    """
    L = L.expand()

    if isinstance(L, SuperOperatorPlus):
        spres = []
        sposts = []
        collapse_form = defaultdict(lambda: defaultdict(int))
        for s in L.operands:
            if isinstance(s, ScalarTimesSuperOperator):
                coeff, term = s.operands
            else:
                coeff, term = One, s
            if isinstance(term, SPre):
                spres.append(coeff * term.operands[0])
            elif isinstance(term, SPost):
                sposts.append((coeff * term.operands[0]))
            else:
                if (not isinstance(term, SuperOperatorTimes) or not
                        len(term.operands) == 2 or not
                        (isinstance(term.operands[0], SPre) and
                        isinstance(term.operands[1], SPost))):
                    raise BadLiouvillianError(
                            "All terms of the Liouvillian need to be of form "
                            "SPre(X), SPost(X) or SPre(X)*SPost(X): This term "
                            "is in violation {!s}".format(term))
                spreL, spostL = term.operands
                Li, Ljd = spreL.operands[0], spostL.operands[0]

                try:
                    complex(coeff)
                except (ValueError, TypeError):
                    symbolic = True
                    coeff = coeff.simplify_scalar()

                collapse_form[Li][Ljd] = coeff

        basis = sorted(collapse_form.keys())

        warn_msg = ("Warning: the Liouvillian is probably malformed: "
                    "The coefficients of SPre({!s})*SPost({!s}) and "
                    "SPre({!s})*SPost({!s}) should be complex conjugates "
                    "of each other")
        for ii, Li in enumerate(basis):
            for Lj in basis[ii:]:
                cij = collapse_form[Li][Lj.adjoint()]
                cji = collapse_form[Lj][Li.adjoint()]
                if cij !=0 or cji !=0:
                    diff = (cij.conjugate() - cji)
                    try:
                        diff = complex(diff)
                        if abs(diff) > 1e-6:
                            print(warn_msg.format(Li, Lj.adjoint(), Lj,
                                                  Li.adjoint()))
                    except ValueError:
                        symbolic = True
                        if diff.simplify():
                            print("Warning: the Liouvillian my be malformed, "
                                  "convert to numerical representation")
        final_Lis = []
        if symbolic:
            if len(basis) == 1:
                l1 = basis[0]
                kappa1 = collapse_form[l1][l1.adjoint()]
                final_Lis = [sqrt(kappa1) * l1]
                sdiff = (l1.adjoint() * l1 * kappa1 / 2)
                spres.append(sdiff)
                sposts.append(sdiff)
#            elif len(basis) == 2:
#                l1, l2 = basis
#                kappa_1 = collapse_form[l1][l1.adjoint()]
#                kappa_2 = collapse_form[l2][l2.adjoint()]
#                kappa_12 = collapse_form[l1][l2.adjoint()]
#                kappa_21 = collapse_form[l2][l1.adjoint()]
##                assert (kappa_12.conjugate() - kappa_21) == 0
            else:
                M = SympyMatrix(len(basis), len(basis),
                                lambda i,j: collapse_form[basis[i]][basis[j]
                                            .adjoint()])

                # First check if M is already diagonal (sympy does not handle
                # this well, for some reason)
                diag = True
                for i in range(len(basis)):
                    for j in range(i):
                        if M[i,j].apply_rules() != 0 or M[j, i].apply_rules != 0:
                            diag = False
                            break
                    if not diag:
                        break
                if diag:
                    for bj in basis:
                        final_Lis.append(
                                bj * sqrt(collapse_form[bj][bj.adjoint()]))
                        sdiff = (bj.adjoint() * bj *
                                 collapse_form[bj][bj.adjoint()]/2)
                        spres.append(sdiff)
                        sposts.append(sdiff)

                # Try sympy algo
                else:
                    try:
                        data = M.eigenvects()

                        for evalue, multiplicity, ebasis in data:
                            if not evalue:
                                continue
                            for b in ebasis:
                                new_L = (sqrt(evalue) * sum(cj[0] * Lj
                                         for (cj, Lj)
                                         in zip(b.tolist(), basis))).expand()
                                final_Lis.append(new_L)
                                sdiff = (new_L.adjoint() * new_L / 2).expand()
                                spres.append(sdiff)
                                sposts.append(sdiff)

                    except NotImplementedError:
                        raise CannotSymbolicallyDiagonalize((
                            "The matrix {} is too hard to diagonalize "
                            "symbolically. Please try converting to fully "
                            "numerical representation.").format(M))
        else:
            M = np_array([[complex(collapse_form[Li][Lj.adjoint()])
                           for Lj in basis] for Li in basis])

            vals, vecs = eigh(M)
            for sv, vec in zip(np_sqrt(vals), vecs.transpose()):
                new_L = sum((sv * ci) * Li for (ci, Li) in zip(vec, basis))
                final_Lis.append(new_L)
                sdiff = (.5 * new_L.adjoint()*new_L).expand()
                spres.append(sdiff)
                sposts.append(sdiff)

        miHspre = sum(spres)
        iHspost = sum(sposts)

        if ((not (miHspre + iHspost) is ZeroOperator) or not
                (miHspre.adjoint() + miHspre) is ZeroOperator):
            print("Warning, potentially malformed Liouvillian {!s}".format(L))

        final_H = (I*miHspre).expand()
        return final_H, final_Lis

    else:
        if L is ZeroSuperOperator:
            return ZeroOperator, []

        raise BadLiouvillianError(str(L))


SuperOperator._zero = ZeroSuperOperator
SuperOperator._one = IdentitySuperOperator
SuperOperator._base_cls = SuperOperator
SuperOperator._scalar_times_expr_cls = ScalarTimesSuperOperator
SuperOperator._plus_cls = SuperOperatorPlus
SuperOperator._times_cls = SuperOperatorTimes
SuperOperator._adjoint_cls = SuperAdjoint
SuperOperator._indexed_sum_cls = None  # TODO
SuperOperator._indexed_sum_cls = None  # TODO
SuperOperator._derivative_cls = SuperOperatorDerivative
