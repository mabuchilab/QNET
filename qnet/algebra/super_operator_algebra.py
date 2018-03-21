# coding=utf-8
# This file is part of QNET.
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

"""
The specification of a quantum mechanics symbolic super-operator algebra.
See :ref:`super_operator_algebra` for more details.
"""
import re
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import product as cartesian_product
from collections import defaultdict, OrderedDict

from numpy.linalg import eigh
from numpy import (sqrt as np_sqrt, array as np_array)

from sympy import (Matrix as SympyMatrix, sqrt, I)

from .scalar_types import SCALAR_TYPES
from .abstract_algebra import (
    Operation, Expression, AlgebraError, assoc, orderby,
    filter_neutral, match_replace, match_replace_binary, AlgebraException,
    check_rules_dict, ScalarTimesExpression)
from .singleton import Singleton, singleton_object
from .pattern_matching import wc, pattern_head, pattern
from .hilbert_space_algebra import TrivialSpace, LocalSpace, ProductSpace
from .operator_algebra import (
    Operator, sympyOne, ScalarTimesOperator, OperatorPlus, ZeroOperator,
    IdentityOperator, simplify_scalar)
from .ordering import (
    KeyTuple, FullCommutativeHSOrder, DisjunctCommutativeHSOrder)
from .matrix_algebra import Matrix


__all__ = [
    'BadLiouvillianError', 'CannotSymbolicallyDiagonalize', 'SPost',
    'SPre', 'ScalarTimesSuperOperator', 'SuperAdjoint',
    'SuperCommutativeHSOrder', 'SuperOperator', 'SuperOperatorOperation',
    'SuperOperatorPlus', 'SuperOperatorSymbol', 'SuperOperatorTimes',
    'SuperOperatorTimesOperator', 'anti_commutator', 'commutator',
    'lindblad', 'liouvillian', 'liouvillian_normal_form',
    'IdentitySuperOperator', 'ZeroSuperOperator']

__private__ = []  # anything not in __all__ must be in __private__


###############################################################################
# Exceptions
###############################################################################


class CannotSymbolicallyDiagonalize(AlgebraException):
    pass


class BadLiouvillianError(AlgebraError):
    """Raise when a Liouvillian is not of standard Lindblad form."""
    pass


###############################################################################
# Abstract base classes
###############################################################################


class SuperOperator(metaclass=ABCMeta):
    """The super-operator abstract base class.

    Any super-operator contains an associated HilbertSpace object,
    on which it is taken to act non-trivially.
    """

    @abstractproperty
    def space(self):
        """The Hilbert space associated with the operator on which it acts
        non-trivially"""
        raise NotImplementedError(self.__class__.__name__)

    def superadjoint(self):
        """The super-operator adjoint (w.r.t to the ``Tr`` operation).
        See :py:class:`SuperAdjoint` documentation.

        :return: The super-adjoint of the super-operator.
        :rtype: SuperOperator
        """
        return SuperAdjoint.create(self)

    def expand(self):
        """Expand out distributively all products of sums. Note that this does
        not expand out sums of scalar coefficients.

        :return: A fully expanded sum of superoperators.
        :rtype: SuperOperator
        """
        return self._expand()

    def simplify_scalar(self):
        """
        Simplify all scalar coefficients within the Operator expression.

        :return: The simplified expression.
        :rtype: Operator
        """
        return self._simplify_scalar()

    def _simplify_scalar(self):
        return self

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

    def __add__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return SuperOperatorPlus.create(self,
                                            other * IdentitySuperOperator)
        elif isinstance(other, SuperOperator):
            return SuperOperatorPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return ScalarTimesSuperOperator.create(other, self)
        elif isinstance(other, Operator):
            return SuperOperatorTimesOperator.create(self, other)
        elif isinstance(other, SuperOperator):
            return SuperOperatorTimes.create(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return ScalarTimesSuperOperator.create(other, self)
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
        return NotImplemented


class SuperOperatorOperation(SuperOperator, Operation, metaclass=ABCMeta):
    """Base class for Operations acting only on SuperOperator arguments."""

    def __init__(self, *operands):
        op_spaces = [o.space for o in operands]
        self._space = ProductSpace.create(*op_spaces)
        self._order_key = KeyTuple([
            '~'+self.__class__.__name__, '__',
            1.0] + [op._order_key for op in operands])
        super().__init__(*operands)

    @property
    def space(self):
        return self._space

    def _simplify_scalar(self):
        return self.create(*[o.simplify_scalar() for o in self.operands])

    @abstractmethod
    def _expand(self):
        raise NotImplementedError()


###############################################################################
# Superoperator algebra elements
###############################################################################


class SuperOperatorSymbol(SuperOperator, Expression):
    """Super-operator symbol class, parametrized by an identifier string and an
    associated Hilbert space.

    Instantiate as::

        SuperOperatorSymbol(name, hs)

    :param label: Symbol identifier
    :type label: str
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace
    """
    _rx_label = re.compile('^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9().+-]+)?$')

    def __init__(self, label, *, hs):
        if not self._rx_label.match(label):
            raise ValueError("label '%s' does not match pattern '%s'"
                                % (label, self._rx_label.pattern))
        self._label = label
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        elif isinstance(hs, tuple):
            hs = ProductSpace.create(*[LocalSpace(h) for h in hs])
        self._hs = hs
        self._order_key = KeyTuple((self.__class__.__name__, str(label),
                                    1.0))
        super().__init__(label, hs=hs)

    @property
    def label(self):
        return self._label

    @property
    def args(self):
        return (self.label, )

    @property
    def kwargs(self):
        return {'hs': self._hs}

    @property
    def space(self):
        return self._hs

    def _expand(self):
        return self

    def all_symbols(self):
        return {self}


@singleton_object
class IdentitySuperOperator(SuperOperator, Expression, metaclass=Singleton):
    """IdentitySuperOperator constant (singleton) object."""

    @property
    def space(self):
        return TrivialSpace

    @property
    def _order_key(self):
        return KeyTuple(('~~', self.__class__.__name__, 1.0))

    @property
    def args(self):
        return tuple()

    def _superadjoint(self):
        return self

    def _expand(self):
        return self

    def __eq__(self, other):
        return self is other or other == 1

    def all_symbols(self):
        return set(())


@singleton_object
class ZeroSuperOperator(SuperOperator, Expression, metaclass=Singleton):
    """ZeroSuperOperator constant (singleton) object."""

    @property
    def space(self):
        return TrivialSpace

    @property
    def _order_key(self):
        return KeyTuple(('~~', self.__class__.__name__, 1.0))

    @property
    def args(self):
        return tuple()

    def _superadjoint(self):
        return self

    def _expand(self):
        return self

    def __eq__(self, other):
        return self is other or other == 0

    def all_symbols(self):
        return set(())


###############################################################################
# Algebra Operations
###############################################################################


class SuperOperatorPlus(SuperOperatorOperation):
    """A sum of super-operators.

    Instantiate as::

        OperatorPlus(*summands)

    :param SuperOperator summands: super-operator summands.
    """
    neutral_element = ZeroSuperOperator
    _binary_rules = OrderedDict()  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary]

    order_key = FullCommutativeHSOrder

    def _expand(self):
        return sum((o.expand() for o in self.operands), ZeroSuperOperator)
        # Note that `SuperOperatorPlus(*[o.expand() for o in self.operands])`
        # does not give a sufficiently simplified result in this case


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


class SuperOperatorTimes(SuperOperatorOperation):
    """A product of super-operators that denotes order of application of
    super-operators (right to left)::

        SuperOperatorTimes(*factors)

    :param SuperOperator factors: Super-operator factors.
    """
    neutral_element = IdentitySuperOperator
    _binary_rules = OrderedDict()  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary]

    order_key = SuperCommutativeHSOrder

    @classmethod
    def create(cls, *ops):
        if any(o == ZeroSuperOperator for o in ops):
            return ZeroSuperOperator
        return super().create(*ops)

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands
                        if isinstance(eo, SuperOperatorPlus) else (eo,)
                        for eo in eops]
        # iterate over a Cartesian product of all factor summands, form product
        # of each tuple and sum over result
        return sum((SuperOperatorTimes.create(*combo)
                    for combo in cartesian_product(*eopssummands)),
                   ZeroSuperOperator)


class ScalarTimesSuperOperator(SuperOperator, ScalarTimesExpression):
    """Multiply an operator by a scalar coefficient::

        ScalarTimesSuperOperator(coeff, term)

    :param coeff: Scalar coefficient.
    :type coeff: SCALAR_TYPES
    :param term: The super-operator that is multiplied.
    :type term: SuperOperator
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, ]

    @property
    def space(self):
        return self.term.space

    @property
    def _order_key(self):
        from qnet.printing import ascii
        t = self.term._order_key
        try:
            c = abs(float(self.coeff))  # smallest coefficients first
        except (ValueError, TypeError):
            c = float('inf')
        return KeyTuple(t[:2] + (c, ) + t[3:] + (ascii(self.coeff), ))

    def _expand(self):
        c, t = self.coeff, self.term
        et = t.expand()
        if isinstance(et, SuperOperatorPlus):
            return sum((c * eto for eto in et.operands), ZeroSuperOperator)
        return c * et

    def _simplify_scalar(self):
        coeff, term = self.operands
        return simplify_scalar(coeff) * term.simplify_scalar()

#    def _pseudo_inverse(self):
#        c, t = self.operands
#        return t.pseudo_inverse() / c

    def __complex__(self):
        if self.term is IdentitySuperOperator:
            return complex(self.coeff)
        return NotImplemented

    def __float__(self):
        if self.term is IdentitySuperOperator:
            return float(self.coeff)
        return NotImplemented


class SuperAdjoint(SuperOperatorOperation):
    r"""The symbolic SuperAdjoint of a super-operator.

    The math notation for this is typically

    .. math::
        {\rm SuperAdjoint}(\mathcal{L}) =: \mathcal{L}^*

    and for any super operator :math:`\mathcal{L}`, its super-adjoint
    :math:`\mathcal{L}^*` satisfies for any pair of operators :math:`M,N`:

    .. math::
        {\rm Tr}[M (\mathcal{L}N)] = Tr[(\mathcal{L}^*M)  N]


    :param L: The super-operator to take the adjoint of.
    :type L: SuperOperator
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, ]

    def __init__(self, operand):
        super().__init__(operand)
        self._order_key = (self.operands[0]._order_key +
                           KeyTuple((self.__class__.__name__, )))

    @property
    def operand(self):
        return self.operands[0]

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, SuperOperatorPlus):
            return sum((eoo.superadjoint() for eoo in eo.operands),
                       ZeroSuperOperator)
        return eo._superadjoint()


class SPre(SuperOperator, Operation):
    """Linear pre-multiplication operator.

    Acting ``SPre(A)`` on an operator ``B`` just yields the product ``A * B``
    """

    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, ]

    def __init__(self, op):
        self._order_key = KeyTuple(("AAA" + self.__class__.__name__,
                                    '__', 1.0, op._order_key))
        # The prepended "AAA" ensures that SPre sorts before SPost
        super().__init__(op)

    @property
    def space(self):
        return self.operands[0].space

    def _expand(self):
        oe = self.operands[0].expand()
        if isinstance(oe, OperatorPlus):
            return sum(SPre.create(oet) for oet in oe.operands)
        return SPre.create(oe)

    def _simplify_scalar(self):
        return self.create(self.operands[0].simplify_scalar())


class SPost(SuperOperator, Operation):
    """Linear post-multiplication operator.

        Acting ``SPost(A)`` on an operator ``B`` just yields the reversed
        product ``B * A``.
    """

    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, ]

    def __init__(self, op):
        self._order_key = KeyTuple(("BBB" + self.__class__.__name__,
                                    '__', 1.0, op._order_key))
        # The prepended "BBB" ensures that SPre sorts before SPost
        super().__init__(op)

    @property
    def space(self):
        return self.operands[0].space

    def _expand(self):
        oe = self.operands[0].expand()
        if isinstance(oe, OperatorPlus):
            return sum(SPost.create(oet) for oet in oe.operands)
        return SPost.create(oe)

    def _simplify_scalar(self):
        return self.create(self.operands[0].simplify_scalar())


class SuperOperatorTimesOperator(Operator, Operation):
    """Application of a super-operator to an operator (result is an Operator).

    :param SuperOperator sop: The super-operator to apply.
    :param Operator op: The operator it is applied to.
    """
    _rules = OrderedDict()  # see end of module
    _simplifications = [match_replace, ]

    def __init__(self, sop, op):
        assert isinstance(sop, SuperOperator)
        assert isinstance(op, Operator)
        self._order_key = KeyTuple(
                (self.__class__.__name__, '__',
                 1.0, sop._order_key, op._order_key))
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

    def _simplify_scalar(self):
        sop, op = self.sop, self.op
        return sop.simplify_scalar() * op.simplify_scalar()


###############################################################################
# Constructor Routines
###############################################################################


def commutator(A, B = None):
    """If ``B != None``, return the commutator :math:`[A,B]`, otherwise return
    the super-operator :math:`[A,\cdot]`.  The super-operator :math:`[A,\cdot]`
    maps any other operator ``B`` to the commutator :math:`[A, B] = A B - B A`.

    :param Operator A: The first operator to form the commutator of.
    :param (Operator or None) B: The second operator to form the commutator of, or None.
    :return: The linear superoperator :math:`[A,\cdot]`
    :rtype: SuperOperator
    """
    if B:
        return A * B - B * A
    return SPre(A) - SPost(A)

def anti_commutator(A, B = None):
    """If ``B != None``, return the anti-commutator :math:`\{A,B\}`, otherwise
    return the super-operator :math:`\{A,\cdot\}`.  The super-operator
    :math:`\{A,\cdot\}` maps any other operator ``B`` to the anti-commutator
    :math:`\{A, B\} = A B + B A`.

    :param Operator A: The first operator to form all anti-commutators of.
    :param (Operator or None) B: The second operator to form the anti-commutator of, or None.
    :return: The linear superoperator :math:`[A,\cdot]`
    :rtype: SuperOperator
    """
    if B:
        return A * B + B * A
    return SPre(A) + SPost(A)

def lindblad(C):
    """Return ``SPre(C) * SPost(C.adjoint()) - (1/2) *
    santi_commutator(C.adjoint()*C)``.  These are the super-operators
    :math:`\mathcal{D}[C]` that form the collapse terms of a Master-Equation.
    Applied to an operator :math:`X` they yield

    .. math::
        \mathcal{D}[C] X = C X C^\dagger - {1\over 2} (C^\dagger C X + X C^\dagger C)

    :param C: The associated collapse operator
    :type C: Operator
    :return: The Lindblad collapse generator.
    :rtype: SuperOperator
    """
    if isinstance(C, SCALAR_TYPES):
        return ZeroSuperOperator
    return SPre(C) * SPost(C.adjoint()) - (sympyOne/2) * anti_commutator(C.adjoint() * C)


def liouvillian(H, Ls = []):
    r"""Return the Liouvillian super-operator associated with a Hamilton
    operator ``H`` and a set of collapse-operators ``Ls = [L1, L2, ...]``.

    The Liouvillian :math:`\mathcal{L}` generates the Markovian-dynamics of a
    system via the Master equation:

    .. math::
        \dot{\rho} = \mathcal{L}\rho = -i[H,\rho] + \sum_{j=1}^n \mathcal{D}[L_j] \rho

    :param H: The associated Hamilton operator
    :type H: Operator
    :param Ls: A sequence of collapse operators.
    :type Ls: sequence or Matrix
    :return: The Liouvillian super-operator.
    :rtype: SuperOperator
    """
    if isinstance(Ls, Matrix):
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

    >>> from sympy import symbols
    >>> from qnet.algebra import Create, Destroy
    >>> kappa_1, kappa_2 = symbols('kappa_1, kappa_2', positive = True)
    >>> Delta = symbols('Delta', real = True)
    >>> alpha = symbols('alpha')
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

    :param L: The Liouvillian
    :type L: SuperOperator
    :return: ``(H, Ls)``
    :rtype: tuple
    :raises: BadLiouvillianError
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
                coeff, term = sympyOne, s
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
                    coeff = coeff.simplify()

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
                        if M[i,j].simplify() != 0 or M[j,i].simplify != 0:
                            diag = False
                            break
                    if diag == False:
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


###############################################################################
# Algebraic rules
###############################################################################


def _algebraic_rules():
    u = wc("u", head=SCALAR_TYPES)
    v = wc("v", head=SCALAR_TYPES)

    A = wc("A", head=Operator)
    B = wc("B", head=Operator)
    C = wc("C", head=Operator)

    sA = wc("sA", head=SuperOperator)
    sA__ = wc("sA__", head=SuperOperator)
    sB = wc("sB", head=SuperOperator)

    sA_plus = wc("sA", head=SuperOperatorPlus)
    sA_times = wc("sA", head=SuperOperatorTimes)

    ScalarTimesSuperOperator._rules.update(check_rules_dict([
        ('one', (
            pattern_head(1, sA),
            lambda sA: sA)),
        ('zero', (
            pattern_head(0, sA),
            lambda sA: ZeroSuperOperator)),
        ('zeroSop', (
            pattern_head(u, ZeroSuperOperator),
            lambda u: ZeroSuperOperator)),
        ('uvSop', (
            pattern_head(u, pattern(ScalarTimesSuperOperator, v, sA)),
            lambda u, v, sA: (u * v) * sA)),
    ]))

    SuperOperatorPlus._binary_rules.update(check_rules_dict([
        ('upv', (
            pattern_head(
                pattern(ScalarTimesSuperOperator, u, sA),
                pattern(ScalarTimesSuperOperator, v, sA)),
            lambda u, v, sA: (u + v) * sA)),
        ('up1', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sA),
            lambda u, sA: (u + 1) * sA)),
        ('1pv', (
            pattern_head(sA, pattern(ScalarTimesSuperOperator, v, sA)),
            lambda v, sA: (1 + v) * sA)),
        ('two', (
            pattern_head(sA, sA),
            lambda sA: 2 * sA)),
    ]))

    SuperOperatorTimes._binary_rules.update(check_rules_dict([
        ('uSaSb1', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sB),
            lambda u, sA, sB: u * (sA * sB))),
        ('uSaSb2', (
            pattern_head(sA, pattern(ScalarTimesSuperOperator, u, sB)),
            lambda sA, u, sB: u * (sA * sB))),
        ('spre', (
            pattern_head(pattern(SPre, A), pattern(SPre, B)),
            lambda A, B: SPre.create(A*B))),
        ('spost', (
            pattern_head(pattern(SPost, A), pattern(SPost, B)),
            lambda A, B: SPost.create(B*A))),
    ]))

    SuperAdjoint._rules.update(check_rules_dict([
        ('uSa', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA)),
            lambda u, sA: u * sA.superadjoint())),
        ('plus', (
            pattern_head(sA_plus),
            lambda sA: SuperOperatorPlus.create(
                *[o.superadjoint() for o in sA.operands]))),
        ('times', (
            pattern_head(sA_times),
            lambda sA: SuperOperatorTimes.create(
                *[o.superadjoint() for o in sA.operands[::-1]]))),
        ('adjoint', (
            pattern_head(pattern(SuperAdjoint, sA)),
            lambda sA: sA)),
        ('spre', (
            pattern_head(pattern(SPre, A)),
            lambda A: SPost.create(A))),
        ('spost', (
            pattern_head(pattern(SPost, A)),
            lambda A: SPre.create(A))),
        ('identity', (
            pattern_head(IdentitySuperOperator),
            lambda: IdentitySuperOperator)),
        ('rule', (
            pattern_head(ZeroSuperOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SPre._rules.update(check_rules_dict([
        ('scal', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u * SPre.create(A))),
        ('id', (
            pattern_head(IdentityOperator),
            lambda: IdentitySuperOperator)),
        ('zero', (
            pattern_head(ZeroOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SPost._rules.update(check_rules_dict([
        ('scal', (
            pattern_head(pattern(ScalarTimesOperator, u, A)),
            lambda u, A: u * SPost.create(A))),
        ('id', (
            pattern_head(IdentityOperator),
            lambda: IdentitySuperOperator)),
        ('zero', (
            pattern_head(ZeroOperator),
            lambda: ZeroSuperOperator)),
    ]))

    SuperOperatorTimesOperator._rules.update(check_rules_dict([
        ('plus', (
            pattern_head(sA_plus, B),
            lambda sA, B:
                SuperOperatorPlus.create(*[o*B for o in sA.operands]))),
        ('id', (
            pattern_head(IdentitySuperOperator, B),
            lambda B: B)),
        ('zero', (
            pattern_head(ZeroSuperOperator, B),
            lambda B: ZeroOperator)),
        ('uSaSb1', (
            pattern_head(pattern(ScalarTimesSuperOperator, u, sA), B),
            lambda u, sA, B: u * (sA * B))),
        ('uSaSb2', (
            pattern_head(sA, pattern(ScalarTimesOperator, u, B)),
            lambda u, sA, B: u * (sA * B))),
        ('sAsBC', (
            pattern_head(sA, pattern(SuperOperatorTimesOperator, sB, C)),
            lambda sA, sB, C: (sA * sB) * C)),
        ('AB', (
            pattern_head(pattern(SPre, A), B),
            lambda A, B: A*B)),
        ('xxx1', (  # XXX
            pattern_head(
                pattern(SuperOperatorTimes, sA__, pattern(SPre, B)), C),
            lambda sA, B, C: (
                SuperOperatorTimes.create(*sA) * (pattern(SPre, B) * C)))),
        ('spost', (
            pattern_head(pattern(SPost, A), B),
            lambda A, B: B*A)),
        ('xxx2', (  # XXX
            pattern_head(
                pattern(SuperOperatorTimes, sA__, pattern(SPost, B)), C),
            lambda sA, B, C: (
                SuperOperatorTimes.create(*sA) * (pattern(SPost, B) * C)))),
    ]))


_algebraic_rules()
