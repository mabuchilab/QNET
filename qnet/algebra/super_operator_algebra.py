# coding=utf-8
#This file is part of QNET.
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
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

"""
Super-Operator Algebra
======================

The specification of a quantum mechanics symbolic super-operator algebra.
See :ref:`super_operator_algebra` for more details.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import product as cartesian_product
from collections import defaultdict

from numpy.linalg import eigh
from numpy import (sqrt as np_sqrt, array as np_array)

from sympy import (
        symbols, Basic as SympyBasic, Add, Matrix as SympyMatrix, sqrt, I)

from .abstract_algebra import (
        Operation, Expression, singleton, tex, AlgebraError,
        assoc, orderby, filter_neutral, match_replace, match_replace_binary,
        AlgebraException, KeyTuple, cache_attr)
from .pattern_matching import wc, pattern_head, pattern
from .hilbert_space_algebra import (
        HilbertSpace, FullSpace, TrivialSpace, LocalSpace, ProductSpace)
from .operator_algebra import (
        Operator, sympyOne, ScalarTimesOperator, OperatorPlus, ZeroOperator,
        IdentityOperator, identifier_to_tex, simplify_scalar, OperatorSymbol,
        Matrix, Create, Destroy)


class SuperOperator(metaclass=ABCMeta):
    """The super-operator abstract base class.

    Any super-operator contains an associated HilbertSpace object,
    on which it is taken to act non-trivially.
    """

    # which data types may serve as scalar coefficients
    scalar_types = Operator.scalar_types

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
        if isinstance(other, SuperOperator.scalar_types):
            return SuperOperatorPlus.create(self,
                                            other * IdentitySuperOperator)
        elif isinstance(other, SuperOperator):
            return SuperOperatorPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, SuperOperator.scalar_types):
            return ScalarTimesSuperOperator.create(other, self)
        elif isinstance(other, Operator):
            return SuperOperatorTimesOperator.create(self, other)
        elif isinstance(other, SuperOperator):
            return SuperOperatorTimes.create(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, SuperOperator.scalar_types):
            return ScalarTimesSuperOperator.create(other, self)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, SuperOperator.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented


class SuperOperatorSymbol(SuperOperator, Expression):
    """Super-operator symbol class, parametrized by an identifier string and an
    associated Hilbert space.

    Instantiate as::

        SuperOperatorSymbol(name, hs)

    :param name: Symbol identifier
    :type name: str
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace
    """
    signature = (str, (HilbertSpace, str, int, tuple)), {}

    def __init__(self, label, hs):
        self.label = label
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        elif isinstance(hs, tuple):
            hs = ProductSpace.create(*[LocalSpace(h) for h in hs])
        self._hs = hs
        super().__init__(label, hs)

    @property
    def args(self):
        return self.label, self._hs

    def __str__(self):
        return self.label

    @cache_attr('_tex')
    def tex(self):
        return r"\hat{{{}}}".format(identifier_to_tex(self.label))

    @property
    def space(self):
        return self._hs

    def _expand(self):
        return self

    def all_symbols(self):
        return {self}


@singleton
class IdentitySuperOperator(SuperOperator, Expression):
    """IdentitySuperOperator constant (singleton) object."""

    @property
    def space(self):
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def _superadjoint(self):
        return self

    def _expand(self):
        return self

    def tex(self):
        return r"\hat{1}"

    def __str__(self):
        return "_1_"

    def __eq__(self, other):
        return self is other or other == 1

    def all_symbols(self):
        return set(())


@singleton
class ZeroSuperOperator(SuperOperator, Expression):
    """ZeroSuperOperator constant (singleton) object."""

    @property
    def space(self):
        return TrivialSpace

    @property
    def args(self):
        return tuple()

    def _superadjoint(self):
        return self

    def _expand(self):
        return self

    def tex(self):
        return r"\hat{0}"

    def __eq__(self, other):
        return self is other or other == 0

    def __str__(self):
        return "_0_"

    def all_symbols(self):
        return set(())


class SuperOperatorOperation(SuperOperator, Operation):
    """Base class for Operations acting only on SuperOperator arguments."""
    signature = (SuperOperator, {})

    @property
    def space(self):
        return ProductSpace.create(*(o.space for o in self.operands))

    def _simplify_scalar(self):
        return self.__class__.create(*[o.simplify_scalar()
                                       for o in self.operands])

    def _expand(self):
        raise NotImplementedError()


class SuperOperatorPlus(SuperOperatorOperation):
    """A sum of super-operators.

    Instantiate as::

        OperatorPlus(*summands)

    :param SuperOperator summands: super-operator summands.
    """
    neutral_element = ZeroSuperOperator
    _binary_rules = []  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary,
                        filter_neutral]

    @classmethod
    def order_key(cls, a):
        if isinstance(a, ScalarTimesSuperOperator):
            c = a.coeff
            if isinstance(c, SympyBasic):
                c = str(c)
            return KeyTuple((Expression.order_key(a.term), c))
        return KeyTuple((Expression.order_key(a), 1))

    def _expand(self):
        return sum((o.expand() for o in self.operands), ZeroSuperOperator)

    @cache_attr('_tex')
    def tex(self):
        ret = self.operands[0].tex()

        for o in self.operands[1:]:
            if (isinstance(o, ScalarTimesSuperOperator) and
                    ScalarTimesOperator.has_minus_prefactor(o.coeff)):
                ret += " - " + tex(-o)
            else:
                ret += " + " + tex(o)
        return ret

    @cache_attr('_str')
    def __str__(self):
        ret = str(self.operands[0])
        for o in self.operands[1:]:
            if (isinstance(o, ScalarTimesSuperOperator) and
                    ScalarTimesOperator.has_minus_prefactor(o.coeff)):
                ret += " - " + str(-o)
            else:
                ret += " + " + str(o)
        return ret


class SuperOperatorOrderKey(object):
    """Auxiliary class that generates the correct pseudo-order relation for
    operator products.  Only operators acting on different Hilbert spaces are
    commuted to achieve the order specified in the full HilbertSpace.  I.e.,
    sorted(factors, key = OperatorOrderKey) achieves this ordering.
    """
    def __init__(self, op):
        if isinstance(op, ScalarTimesSuperOperator):
            op = op.term
        space = op.space
        self.op = op
        self.full = False
        self.trivial = False
        if isinstance(space, LocalSpace):
            self.local_spaces = {space.args, }
        elif space is TrivialSpace:
            self.local_spaces = set(())
            self.trivial = True
        elif space is FullSpace:
            self.full = True
        else:
            assert isinstance(space, ProductSpace)
            self.local_spaces = {s.args for s in space.args}

    def __lt__(self, other):
        if isinstance(self.op, SPre) and isinstance(other.op, SPost):
            return True
        elif isinstance(self.op, SPost) and isinstance(other.op, SPre):
            return False

        if self.trivial and other.trivial:
            return (Expression.order_key(self.op) <
                    Expression.order_key(other.op))

        if self.full or len(self.local_spaces & other.local_spaces):
            return False
        return tuple(self.local_spaces) < tuple(other.local_spaces)

    def __gt__(self, other):
        if isinstance(self.op, SPost) and isinstance(other.op, SPre):
            return True
        elif isinstance(self.op, SPre) and isinstance(other.op, SPost):
            return False

        if self.trivial and other.trivial:
            return (Expression.order_key(self.op) >
                    Expression.order_key(other.op))

        if self.full or len(self.local_spaces & other.local_spaces):
            return False

        return tuple(self.local_spaces) > tuple(other.local_spaces)

    def __eq__(self, other):
        if ((isinstance(self.op, SPost) and isinstance(other.op, SPre)) or
                (isinstance(self.op, SPre) and isinstance(other.op, SPost))):
            return False
        if self.trivial and other.trivial:
            return (Expression.order_key(self.op) ==
                    Expression.order_key(other.op))

        return self.full or len(self.local_spaces & other.local_spaces) > 0


class SuperOperatorTimes(SuperOperatorOperation):
    """A product of super-operators that denotes order of application of
    super-operators (right to left)::

        SuperOperatorTimes(*factors)

    :param SuperOperator factors: Super-operator factors.
    """
    neutral_element = IdentitySuperOperator
    _binary_rules = []  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary,
                        filter_neutral]

    order_key = SuperOperatorOrderKey

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

    @cache_attr('_tex')
    def tex(self):
        ret = self.operands[0].tex()
        for o in self.operands[1:]:
            if isinstance(o, SuperOperatorPlus):
                ret += r" \left({}\right) ".format(tex(o))
            else:
                ret += " {}".format(tex(o))
        return ret

    @cache_attr('_str')
    def __str__(self):
        ret = str(self.operands[0])
        for o in self.operands[1:]:
            if isinstance(o, SuperOperatorPlus):
                ret += r" ({})".format(str(o))
            else:
                ret += " {}".format(str(o))
        return ret


class ScalarTimesSuperOperator(SuperOperator, Operation):
    """Multiply an operator by a scalar coefficient::

        ScalarTimesSuperOperator(coeff, term)

    :param coeff: Scalar coefficient.
    :type coeff: :py:attr:`SuperOperator.scalar_types`
    :param term: The super-operator that is multiplied.
    :type term: SuperOperator
    """
    signature = SuperOperator.scalar_types, SuperOperator
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def space(self):
        return self.term.space

    @property
    def coeff(self):
        """The scalar coefficient."""
        return self.operands[0]

    @property
    def term(self):
        """The super-operator term."""
        return self.operands[1]

    @cache_attr('_tex')
    def tex(self):
        coeff, term = self.operands

        if isinstance(coeff, Add):
            cs = r"\left({}\right)".format(tex(coeff))
        else:
            cs = "{}".format(tex(coeff))

        if term == IdentitySuperOperator:
            ct = ""
        if isinstance(term, SuperOperatorPlus):
            ct = r" \left({}\right)".format(term.tex())
        else:
            ct = r" {}".format(term.tex())

        return cs + ct

    @cache_attr('_str')
    def __str__(self):
        coeff, term = self.operands

        if isinstance(coeff, Add):
            cs = r"({!s})".format(coeff)
        else:
            cs = "{!s}".format(coeff)

        if term == IdentitySuperOperator:
            ct = ""
        if isinstance(term, SuperOperatorPlus):
            ct = r" ({!s})".format(term)
        else:
            ct = r" {!s}".format(term)

        return cs + ct

    def _expand(self):
        c, t = self.operands
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

    def _substitute(self, var_map):
        coeff, term = self.operands
        st = term.substitute(var_map)
        if isinstance(coeff, SympyBasic):
            svar_map = {k:v for k,v in var_map.items() if not isinstance(k,Expression)}
            sc = coeff.subs(svar_map)
        else:
            sc = substitute(coeff, var_map)
        return sc * st



class SuperAdjoint(SuperOperatorOperation):
    r"""The symbolic SuperAdjoint of a super-operator.

    For a super operator ``L`` use as::

        SuperAdjoint(L)

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
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def operand(self):
        return self.operands[0]

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, SuperOperatorPlus):
            return sum((eoo.superadjoint() for eoo in eo.operands),
                       ZeroSuperOperator)
        return eo._superadjoint()

    @cache_attr('_tex')
    def tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^*"

    @cache_attr('_str')
    def __str__(self):
        if isinstance(self.operand, OperatorSymbol):
            return "{}^*".format(str(self.operand))
        return "({})^*".format(str(self.operand))


class SPre(SuperOperator, Operation):
    """Linear pre-multiplication operator.

    Use as::

        SPre(op)

    Acting ``SPre(A)`` on an operator ``B`` just yields the product ``A * B``
    """

    signature = (Operator, ), {}
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def space(self):
        return self.operands[0].space

    @cache_attr('_tex')
    def tex(self):
        return r"{{\rm spre}}\left[{}\right]".format(self.operands[0].tex())

    @cache_attr('_str')
    def __str__(self):
        return r"spre({!s})".format(self.operands[0])

    def _expand(self):
        oe = self.operands[0].expand()
        if isinstance(oe, OperatorPlus):
            return sum(SPre.create(oet) for oet in oe.operands)
        return SPre.create(oe)

    def _simplify_scalar(self):
        return self.__class__.create(self.operands[0].simplify_scalar())


class SPost(SuperOperator, Operation):
    """Linear post-multiplication operator.

        Use as::

            SPost(op)

        Acting ``SPost(A)`` on an operator ``B`` just yields the reversed
        product ``B * A``.
    """

    signature = (Operator,), {}
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def space(self):
        return self.operands[0].space

    @cache_attr('_tex')
    def tex(self):
        return r"{{\rm spost}}\left[{}\right]".format(self.operands[0].tex())

    @cache_attr('_str')
    def __str__(self):
        return r"spost({!s})".format(self.operands[0])

    def _expand(self):
        oe = self.operands[0].expand()
        if isinstance(oe, OperatorPlus):
            return sum(SPost.create(oet) for oet in oe.operands)
        return SPost.create(oe)

    def _simplify_scalar(self):
        return self.__class__.create(self.operands[0].simplify_scalar())


class SuperOperatorTimesOperator(Operator, Operation):
    """Application of a super-operator to an operator (result is an Operator).

    Use as::

        SuperOperatorTimesOperator(sop, op)

    :param SuperOperator sop: The super-operator to apply.
    :param Operator op: The operator it is applied to.
    """
    signature = (SuperOperator, Operator), {}
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def space(self):
        return self.sop.space * self.op.space

    @property
    def sop(self):
        return self.operands[0]

    @property
    def op(self):
        return self.operands[1]

    @cache_attr('_tex')
    def tex(self):
        sop, op = self.operands

        if isinstance(sop, SuperOperatorPlus):
            cs = r" \left({}\right)".format(tex(sop))
        else:
            cs = " {}".format(tex(sop))

        if isinstance(op, SuperOperatorPlus):
            ct = r" \left({}\right)".format(op.tex())
        else:
            ct = r" {}".format(op.tex())

        return cs + ct

    @cache_attr('_str')
    def __str__(self):
        sop, op = self.operands
        if isinstance(sop, SuperOperatorPlus):
            cs = r"({!s})".format((sop))
        else:
            cs = "{}".format(tex(sop))
        if isinstance(op, OperatorPlus):
            ct = r" ({!s})".format(op)
        else:
            ct = r" {}".format(op)
        return (cs + ct).strip()

    def _expand(self):
        sop, op = self.operands
        sope, ope = sop.expand(), op.expand()
        if isinstance(sope, SuperOperatorPlus):
            sopet = sope.operands
        else:
            sopet = sope,
        if isinstance(ope, OperatorPlus):
            opet = ope.operands
        else:
            opet = ope,

        return sum(st * ot for st in sopet for ot in opet)

    def _series_expand(self, param, about, order):
        sop, op = self.operands
        ope = op.series_expand(param, about, order)
        return tuple(sop * opet for opet in ope)

    def _simplify_scalar(self):
        SOp, Op = self.operands
        return SOp.simplify_scalar() * Op.simplify_scalar()

#    def _pseudo_inverse(self):
#        c, t = self.operands
#        return t.pseudo_inverse() / c


## Expression rewriting _rules

u = wc("u", head=Operator.scalar_types)
v = wc("v", head=Operator.scalar_types)

n = wc("n", head=(int, str))
m = wc("m", head=(int, str))
k = wc("k", head=(int, str))

A = wc("A", head=Operator)
A__ = wc("A__", head=Operator)
A___ = wc("A___", head=Operator)
B = wc("B", head=Operator)
B__ = wc("B__", head=Operator)
B___ = wc("B___", head=Operator)
C = wc("C", head=Operator)

sA = wc("sA", head=SuperOperator)
sA__ = wc("sA__", head=SuperOperator)
sA___ = wc("sA___", head=SuperOperator)
sB = wc("sB", head=SuperOperator)
sB__ = wc("sB__", head=SuperOperator)
sB___ = wc("sB___", head=SuperOperator)


sA_plus = wc("sA", head=SuperOperatorPlus)
sA_times = wc("sA", head=SuperOperatorTimes)


ScalarTimesSuperOperator._rules += [
    (pattern_head(1, sA),
        lambda sA: sA),
    (pattern_head(0, sA),
        lambda sA: ZeroSuperOperator),
    (pattern_head(u, ZeroSuperOperator),
        lambda u: ZeroSuperOperator),
    (pattern_head(u, pattern(ScalarTimesSuperOperator, v, sA)),
        lambda u, v, sA: (u * v) * sA),
]

SuperOperatorPlus._binary_rules += [
    (pattern_head(pattern(ScalarTimesSuperOperator, u, sA),
                  pattern(ScalarTimesSuperOperator, v, sA)),
        lambda u, v, sA: (u + v) * sA),
    (pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sA),
        lambda u, sA: (u + 1) * sA),
    (pattern_head(sA, pattern(ScalarTimesSuperOperator, v, sA)),
        lambda v, sA: (1 + v) * sA),
    (pattern_head(sA, sA),
        lambda sA: 2 * sA),
]


SuperOperatorTimes._binary_rules += [
    (pattern_head(pattern(ScalarTimesSuperOperator, u, sA), sB),
        lambda u, sA, sB: u * (sA * sB)),
    (pattern_head(sA, pattern(ScalarTimesSuperOperator, u, sB)),
        lambda sA, u, sB: u * (sA * sB)),
    (pattern_head(pattern(SPre, A), pattern(SPre, B)),
        lambda A, B: SPre.create(A*B)),
    (pattern_head(pattern(SPost, A), pattern(SPost, B)),
        lambda A, B: SPost.create(B*A)),
]

SuperAdjoint._rules += [
    (pattern_head(pattern(ScalarTimesSuperOperator, u, sA)),
        lambda u, sA: u * sA.superadjoint()),
    (pattern_head(sA_plus),
        lambda sA: SuperOperatorPlus.create(*[o.superadjoint()
                                              for o in sA.operands])),
    (pattern_head(sA_times),
        lambda sA: SuperOperatorTimes.create(*[o.superadjoint()
                                               for o in sA.operands[::-1]])),
    (pattern_head(pattern(SuperAdjoint, sA)),
        lambda sA: sA),
    (pattern_head(pattern(SPre, A)),
        lambda A: SPost.create(A)),
    (pattern_head(pattern(SPost, A)),
        lambda A: SPre.create(A)),
    (pattern_head(IdentitySuperOperator),
        lambda: IdentitySuperOperator),
    (pattern_head(ZeroSuperOperator),
        lambda: ZeroSuperOperator),
]

SPre._rules +=[
    (pattern_head(pattern(ScalarTimesOperator, u, A)),
        lambda u, A: u * SPre.create(A)),
    (pattern_head(IdentityOperator),
        lambda: IdentitySuperOperator),
    (pattern_head(ZeroOperator),
        lambda: ZeroSuperOperator),
]

SPost._rules +=[
    (pattern_head(pattern(ScalarTimesOperator, u, A)),
        lambda u, A: u * SPost.create(A)),
    (pattern_head(IdentityOperator),
        lambda: IdentitySuperOperator),
    (pattern_head(ZeroOperator),
        lambda: ZeroSuperOperator),
]


SuperOperatorTimesOperator._rules +=[
    (pattern_head(sA_plus, B),
        lambda sA, B: sum([o*B for o in sA.operands], ZeroOperator)),
    (pattern_head(IdentitySuperOperator, B),
        lambda B: B),
    (pattern_head(ZeroSuperOperator, B),
        lambda B: ZeroOperator),
    (pattern_head(pattern(ScalarTimesSuperOperator, u, sA), B),
        lambda u, sA, B: u * (sA * B)),
    (pattern_head(sA, pattern(ScalarTimesOperator, u, B)),
        lambda u, sA, B: u * (sA * B)),
    (pattern_head(sA, pattern(SuperOperatorTimesOperator, sB, C)),
        lambda sA, sB, C: (sA * sB) * C),
    (pattern_head(pattern(SPre, A), B),
        lambda A, B: A*B),
    (pattern_head(pattern(SuperOperatorTimes, sA__, pattern(SPre, B)), C),
        lambda sA, B, C: (SuperOperatorTimes.create(*sA) *
                          (pattern(SPre, B) * C))),
    (pattern_head(pattern(SPost, A), B),
        lambda A, B: B*A),
    (pattern_head(pattern(SuperOperatorTimes, sA__, pattern(SPost, B)), C),
        lambda sA, B, C: (SuperOperatorTimes.create(*sA) *
                          (pattern(SPost, B) * C))),
]


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
    if isinstance(C, Operator.scalar_types):
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
        Ls = Ls.matrix.flatten().tolist()
    return -I * commutator(H) + sum((lindblad(L) for L in Ls), ZeroSuperOperator)


class BadLiouvillianError(AlgebraError):
    """Raise when a Liouvillian is not of standard Lindblad form."""
    pass


class CannotSymbolicallyDiagonalize(AlgebraException):
    pass


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

    >>> kappa_1, kappa_2 = symbols('kappa_1, kappa_2', positive = True)
    >>> Delta = symbols('Delta', real = True)
    >>> alpha = symbols('alpha')
    >>> H = (Delta * Create(1) * Destroy(1) +
    ...      (sqrt(kappa_1) / (2 * I)) *
    ...      (alpha * Create(1) - alpha.conjugate() * Destroy(1)))
    >>> Ls = [sqrt(kappa_1) * Destroy(1) + alpha, sqrt(kappa_2) * Destroy(1)]
    >>> LL = liouvillian(H, Ls)
    >>> Hnf, Lsnf = liouvillian_normal_form(LL)
    >>> print(Hnf)
    I*sqrt(kappa_1)*conjugate(alpha) * a₍₁₎ + Delta * (a⁺₍₁₎ * a₍₁₎) - I*alpha*sqrt(kappa_1) * a⁺₍₁₎
    >>> len(Lsnf)
    1
    >>> print(Lsnf[0])
    sqrt(kappa_1 + kappa_2) * a₍₁₎

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
        Ls = []
        spres = []
        sposts = []
        collapse_form = defaultdict(lambda : defaultdict(int))
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
