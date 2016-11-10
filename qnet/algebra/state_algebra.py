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

r"""
State Algebra
=============

This module implements a basic Hilbert space state algebra.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import product as cartesian_product

from sympy import (
        Basic as SympyBasic, Add, series as sympy_series, sqrt, exp, I)

from .abstract_algebra import (
        Operation, Expression, substitute, tex, AlgebraError, assoc, orderby,
        filter_neutral, match_replace, match_replace_binary, KeyTuple,
        CannotSimplify, cache_attr, singleton_object, Singleton)
from .pattern_matching import wc, pattern_head, pattern
from .hilbert_space_algebra import (
        FullSpace, TrivialSpace, LocalSpace, ProductSpace)
from .operator_algebra import (
        Operator, sympyOne, ScalarTimesOperator, OperatorTimes, OperatorPlus,
        IdentityOperator, ZeroOperator, LocalSigma, Create, Destroy, Jplus,
        Jminus, Jz, LocalOperator, Jpjmcoeff, Jzjmcoeff, Jmjmcoeff, Displace,
        Phase)


_TEX_BRA_FMT_SPACE = r'\left\langel{label}\right|_{{space}}'
_STR_BRA_FMT_SPACE = r'<{label}|_{space}'
_TEX_BRA_FMT_NOSPACE = r'\left\langel{label}\right|'
_STR_BRA_FMT_NOSPACE = r'<{label}|'
_TEX_KET_FMT_SPACE = r'\left|{label}\right\rangle_{{space}}'
_STR_KET_FMT_SPACE = r'|{label}>_{space}'
_TEX_KET_FMT_NOSPACE = r'\left|{label}\right\rangle'
_STR_KET_FMT_NOSPACE = r'|{label}>'


class Ket(metaclass=ABCMeta):
    """Basic Ket algebra class to represent Hilbert Space states"""

    scalar_types = Operator.scalar_types

    @abstractproperty
    def space(self):
        """The associated HilbertSpace"""
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """The adjoint of a Ket state, i.e., the corresponding Bra."""
        return Bra(self)

    dag = adjoint

    def expand(self):
        """Expand out distributively all products of sums. Note that this does
        not expand out sums of scalar coefficients.

        :return: A fully expanded sum of states.
        :rtype: Ket
        """
        return self._expand()

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def _str_ket(self):
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def _str_bra(self):
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def _tex_ket(self):
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def _tex_bra(self):
        raise NotImplementedError(self.__class__.__name__)

    @cache_attr('_str')
    def __str__(self):
        return self._str_ket()

    @cache_attr('_tex')
    def tex(self):
        return self._tex_ket()

    def __add__(self, other):
        if isinstance(other, Ket):
            return KetPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Ket.scalar_types):
            return ScalarTimesKet.create(other, self)
        elif isinstance(other, Ket):
            return TensorKet.create(self, other)
        elif isinstance(other, Bra):
            return KetBra.create(self, other.ket)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return OperatorTimesKet.create(other, self)
        elif isinstance(other, Ket.scalar_types):
            return ScalarTimesKet.create(other, self)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __div__(self, other):
        if isinstance(other, Ket.scalar_types):
            return self * (sympyOne / other)
        return NotImplemented

    __truediv__ = __div__


class KetSymbol(Ket, Expression):
    """Ket symbol class, parametrized by an identifier string and an associated
    Hilbert space.

        ``KetSymbol(label, hs)``

    :param label: Symbol identifier
    :type label: str
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace
    """

    def __init__(self, label, hs):
        self.label = label
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        self._hs = hs
        super().__init__(label, hs)

    @property
    def args(self):
        return (self.label, self._hs)

    @property
    def hs(self):
        return self._has

    def _str_ket(self):
        return _STR_KET_FMT_SPACE.format(label=self.label, space=self.space)

    def _str_bra(self):
        return _STR_BRA_FMT_SPACE.format(label=self.label, space=self.space)

    def _tex_ket(self):
        return _TEX_KET_FMT_SPACE.format(label=self.label, space=self.space)

    def _tex_bra(self):
        return _TEX_BRA_FMT_SPACE.format(label=self.label, space=self.space)

    @property
    def space(self):
        return self._hs

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def all_symbols(self):
        return set([self, ])


@singleton_object
class ZeroKet(Ket, Expression, metaclass=Singleton):
    """ZeroKet constant (singleton) object for the null-state."""

    @property
    def space(self):
        return FullSpace

    @property
    def args(self):
        return tuple()

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def _tex_ket(self):
        return "0"

    _tex_bra = _tex_ket

    def __eq__(self, other):
        return self is other or other == 0

    def _str_ket(self):
        return "0"

    _str_bra = _str_ket

    def all_symbols(self):
        return set([])


@singleton_object
class TrivialKet(Ket, Expression, metaclass=Singleton):
    """TrivialKet constant (singleton) object.
    This is the neutral element under the state tensor-product.
    """

    @property
    def space(self):
        return TrivialSpace

    def _adjoint(self):
        return Bra(TrivialKet)

    def _expand(self):
        return self

    @property
    def args(self):
        return tuple()

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def _tex_ket(self):
        return _TEX_KET_FMT_NOSPACE.format(label=r'{\rm id}')

    def _tex_bra(self):
        return _TEX_BRA_FMT_NOSPACE.format(label=r'{\rm id}')

    def __eq__(self, other):
        return self is other or other == 1

    def _str_ket(self):
        return _STR_KET_FMT_NOSPACE.format(label=r'id')

    def _str_bra(self):
        return _STR_BRA_FMT_NOSPACE.format(label=r'id')

    def all_symbols(self):
        return set([])


class LocalKet(Ket, Expression):
    """Base class for atomic (non-composite) ket states of single degrees of
    freedom."""

    def __init__(self, hs, label):
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        self._hs = hs
        self.label = label
        super().__init__(hs, label)

    @property
    def args(self):
        return self._hs, self.label

    @property
    def space(self):
        return self._hs

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def all_symbols(self):
        return {}

    def _tex_ket(self):
        return _TEX_KET_FMT_SPACE.format(label=self.label, space=self.space)

    def _tex_bra(self):
        return _TEX_BRA_FMT_SPACE.format(label=self.label, space=self.space)

    def _str_ket(self):
        return _STR_KET_FMT_SPACE.format(label=self.label, space=self.space)

    def _str_bra(self):
        return _STR_BRA_FMT_SPACE.format(label=self.label, space=self.space)


class BasisKet(LocalKet):
    """Local basis state, labeled by an integer or a string.

    Instantiate as::

        BasisKet(hs, label)

    :param LocalSpace hs: The local Hilbert space degree of freedom.
    :param (str or int) label: The basis state label.
    """
    signature = ((LocalSpace, str, int), (str, int, SympyBasic)), {}



class CoherentStateKet(LocalKet):
    """Local coherent state, labeled by a scalar amplitude.

    Instantiate as::

        CoherentStateKet(hs, amp)

    :param LocalSpace hs: The local Hilbert space degree of freedom.
    :param Ket.scalar_types amp: The coherent displacement amplitude.
    """

    signature = ((LocalSpace, str, int), Ket.scalar_types), {}

    def __init__(self, hs, ampl):
        self.ampl = ampl
        self._str_label = r'D(%s)' % ampl
        self._tex_label = r'D(%s)' % tex(ampl)
        super().__init__(hs, ampl)

    @property
    def args(self):
        return self._hs, self.ampl

    def _tex_ket(self):
        label = self._tex_label
        return _TEX_KET_FMT_SPACE.format(label=label, space=self.space)

    def _tex_bra(self):
        label = self._tex_label
        return _TEX_BRA_FMT_SPACE.format(label=label, space=self.space)

    def _series_expand(self, param, about, order):
        return (self,) + (0,) * (order - 1)

    def _substitute(self, var_map):
        hs, amp = self.space, self.ampl
        if isinstance(amp, SympyBasic):
            svar_map = {k: v for (k, v) in var_map.items()
                        if not isinstance(k, Expression)}
            ampc = amp.subs(svar_map)
        else:
            ampc = substitute(amp, var_map)

        return CoherentStateKet(hs, ampc)

    def all_symbols(self):
        if isinstance(self.ampl, SympyBasic):
            return set([self.ampl, ])
        else:
            return set([])


class UnequalSpaces(AlgebraError):
    pass


def check_kets_same_space(cls, ops, kwargs):
    """Check that all operands are from the same Hilbert space."""
    if not len({o.space for o in ops if o is not ZeroKet}) == 1:
        raise UnequalSpaces(str(ops))
    return ops, kwargs


class KetPlus(Ket, Operation):
    """A sum of Ket states.

    Instantiate as::

        KetPlus(*summands)

    :param summands: State summands.
    :type summands: Ket
    """
    neutral_element = ZeroKet
    _binary_rules = []  # see end of module
    _simplifications = [assoc, orderby, filter_neutral, check_kets_same_space,
                        match_replace_binary, filter_neutral]

    @classmethod
    def order_key(cls, a):
        if isinstance(a, ScalarTimesKet):
            c = a.coeff
            if isinstance(c, SympyBasic):
                c = float('inf')
            return KeyTuple((Expression.order_key(a.term), c))
        return KeyTuple((Expression.order_key(a), 1))

    def _expand(self):
        return sum((o.expand() for o in self.operands), ZeroKet)

    def _series_expand(self, param, about, order):
        res = sum((o.series_expand(param, about, order)
                   for o in self.operands), ZeroKet)
        return res

    def _tex_ketbra(self, bra=False):
        ret = self.operands[0].tex()

        for o in self.operands[1:]:
            if (isinstance(o, ScalarTimesKet) and
                    ScalarTimesOperator.has_minus_prefactor(o.coeff)):
                if bra:
                    ret += " - " + (-o)._tex_bra()
                else:
                    ret += " - " + (-o)._tex_ket()
            else:
                if bra:
                    ret += " + " + o._tex_bra()
                else:
                    ret += " + " + o._tex_ket()
        return ret

    def _tex_ket(self):
        return self._tex_ketbra(bra=False)

    def _tex_bra(self):
        return self._tex_ketbra(bra=True)

    def _str_ketbra(self, bra=False):
        ret = str(self.operands[0])

        for o in self.operands[1:]:
            if (isinstance(o, ScalarTimesKet) and
                    ScalarTimesOperator.has_minus_prefactor(o.coeff)):
                if bra:
                    ret += " - " + (-o)._str_bra()
                else:
                    ret += " - " + (-o)._str_ket()
            else:
                if bra:
                    ret += " + " + o._str_bra()
                else:
                    ret += " + " + o._str_ket()
        return ret

    def _str_ket(self):
        return self._str_ketbra(bra=False)

    def _str_bra(self):
        return self._str_ketbra(bra=True)

    @property
    def space(self):
        return self.operands[0].space


class OverlappingSpaces(AlgebraError):
    pass


class TensorKet(Ket, Operation):
    """A tensor product of kets each belonging to different degrees of freedom.
    Instantiate as::

        TensorKet(*factors)

    :param factors: Ket factors.
    :type factors: Ket
    """
    _binary_rules = []  # see end of module
    neutral_element = TrivialKet
    _simplifications = [assoc, orderby, filter_neutral, match_replace_binary,
                        filter_neutral]

    order_key = OperatorTimes.order_key

    def __init__(self, *operands):
        self._space = None
        super().__init__(*operands)

    @classmethod
    def create(cls, *ops):
        if any(o == ZeroKet for o in ops):
            return ZeroKet
        spc = TrivialSpace
        for o in ops:
            if o.space & spc > TrivialSpace:
                raise OverlappingSpaces(str(ops))
            spc *= o.space
        return super().create(*ops)

    def factor_for_space(self, space):
        """Factor into a Ket defined on the given `space` and a Ket on the
        remaining Hilbert space"""
        if not space <= self.space:
            raise SpaceTooLargeError(str((self, space)))
        if space == self.space:
            return self
        if space is TrivialSpace:
            on_ops = [o for o in self.operands if o.space is TrivialSpace]
            off_ops = [o for o in self.operands if o.space > TrivialSpace]
        else:
            on_ops = [o for o in self.operands
                      if o.space & space > TrivialSpace]
            off_ops = [o for o in self.operands
                       if o.space & space is TrivialSpace]
        return TensorKet.create(*on_ops), TensorKet.create(*off_ops)

    @property
    def space(self):
        if self._space is None:
            self._space = ProductSpace.create(*[o.space
                                                for o in self.operands])
        return self._space

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, KetPlus) else (eo,)
                        for eo in eops]
        # iterate over a Cartesian product of all factor summands, form product
        # of each tuple and sum over result
        return sum((TensorKet.create(*combo)
                    for combo in cartesian_product(*eopssummands)), ZeroKet)

    def _tex_ketbra(self, bra=False):
        if bra:
            ret = self.operands[0]._tex_bra()
        else:
            ret = self.operands[0]._tex_ket()
        for o in self.operands[1:]:
            if isinstance(o, KetPlus):
                if bra:
                    ret += r" \left({}\right) ".format(o._tex_bra())
                else:
                    ret += r" \left({}\right) ".format(o._tex_ket())
            else:
                if bra:
                    ret += " {}".format(o._tex_bra())
                else:
                    ret += " {}".format(o._tex_ket())
        return ret

    def _tex_ket(self):
        return self._tex_ketbra(bra=False)

    def _tex_bra(self):
        return self._tex_ketbra(bra=True)

    def _str_ketbra(self, bra=False):
        if bra:
            ret = self.operands[0]._str_bra()
        else:
            ret = self.operands[0]._str_ket()
        for o in self.operands[1:]:
            if isinstance(o, KetPlus):
                if bra:
                    ret += r" ({}) ".format(o._str_bra())
                else:
                    ret += r" ({}) ".format(o._str_ket())
            else:
                if bra:
                    ret += " {}".format(o._str_bra())
                else:
                    ret += " {}".format(o._str_ket())
        return ret

    def _str_ket(self):
        return self._str_ketbra(bra=False)

    def _str_bra(self):
        return self._str_ketbra(bra=True)


class ScalarTimesKet(Ket, Operation):
    """Multiply a Ket by a scalar coefficient.

    Instantiate as::
        ScalarTimesKet(coefficient, term)

    :param coefficient: Scalar coefficient.
    :type coefficient: Operator.scalar_types
    :param term: The ket that is multiplied.
    :type term: Ket
    """
    signature = (Ket.scalar_types, Ket), {}
    _rules = []  # see end of module
    _simplifications = [match_replace, ]

    @property
    def space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def _tex_ketbra(self, bra=False):
        coeff, term = self.operands

        if isinstance(coeff, Add):
            if bra:
                cs = r" \left({}\right)".format(tex(coeff.conjugate()))
            else:
                cs = r" \left({}\right)".format(tex(coeff))
        else:
            if bra:
                cs = " {}".format(tex(coeff.conjugate()))
            else:
                cs = " {}".format(tex(coeff))

        if isinstance(term, KetPlus):
            if bra:
                ct = r" \left({}\right)".format(term._tex_bra())
            else:
                ct = r" \left({}\right)".format(term._tex_ket())
        else:
            if bra:
                ct = r" {}".format(term._tex_bra())
            else:
                ct = r" {}".format(term._tex_ket())

        return cs + ct

    def _tex_ket(self):
        return self._tex_ketbra(bra=False)

    def _tex_bra(self):
        return self._tex_ketbra(bra=True)

    def _str_ketbra(self, bra=False):
        coeff, term = self.operands

        if isinstance(coeff, Add):
            if bra:
                cs = r" \left({}\right)".format(str(coeff.conjugate()))
            else:
                cs = r" \left({}\right)".format(str(coeff))
        else:
            if bra:
                cs = " {}".format(str(coeff.conjugate()))
            else:
                cs = " {}".format(str(coeff))

        if isinstance(term, KetPlus):
            if bra:
                ct = r" \left({}\right)".format(term._str_bra())
            else:
                ct = r" \left({}\right)".format(term._str_ket())
        else:
            if bra:
                ct = r" {}".format(term._str_bra())
            else:
                ct = r" {}".format(term._str_ket())

        return cs + ct

    def _str_ket(self):
        return self._str_ketbra(bra=False)

    def _str_bra(self):
        return self._str_ketbra(bra=True)

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, KetPlus):
            return sum((c * eto for eto in et.operands), ZeroKet)
        return c * et

    def _series_expand(self, param, about, order):
        ceg = sympy_series(self.coeff, x=param, x0=about, n=None)
        ce = tuple(ceo for ceo, k in zip(ceg, range(order + 1)))
        te = self.term.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k]
                     for n in range(order + 1) for k in range(n + 1))

    def _substitute(self, var_map):
        st = self.term.substitute(var_map)
        if isinstance(self.coeff, SympyBasic):
            svar_map = {k: v for (k, v) in var_map.items()
                        if not isinstance(k, Expression)}
            sc = self.coeff.subs(svar_map)
        else:
            sc = substitute(self.coeff, var_map)
        return sc * st


class SpaceTooLargeError(AlgebraError):
    pass


def check_op_ket_space(cls, ops, kwargs):
    """Check that all operands are from the same Hilbert space."""
    op, ket = ops
    if not op.space <= ket.space:
        raise SpaceTooLargeError(str(op.space) + " <!= " + str(ket.space))
    return ops, kwargs


class OperatorTimesKet(Ket, Operation):
    """Multiply an operator by a scalar coefficient.

    Instantiate as::

        OperatorTimesKet(op, ket)

    :param Operator op: The multiplying operator.
    :param Ket ket: The ket that is multiplied.
    """
    signature = Operator, Ket
    _rules = []  # see end of module
    _simplifications = [match_replace, check_op_ket_space]

    @property
    def space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def _tex_ketbra(self, bra=False):
        coeff, term = self.operands

        if isinstance(coeff, OperatorPlus):
            if bra:
                cs = r"\left({}\right)".format(tex(coeff.adjoint()))
            else:
                cs = r"\left({}\right)".format(tex(coeff))
        else:
            if bra:
                cs = " {}".format(tex(coeff.adjoint()))
            else:
                cs = " {}".format(tex(coeff))

        if isinstance(term, KetPlus):
            if bra:
                ct = r" \left({}\right)".format(term._tex_bra())
            else:
                ct = r" \left({}\right)".format(term._tex_ket())
        else:
            if bra:
                ct = r"{} ".format(term._tex_bra())
            else:
                ct = r" {}".format(term._tex_ket())
        if bra:
            return ct + cs
        else:
            return cs + ct

    def _tex_ket(self):
        return self._tex_ketbra(bra=False)

    def _tex_bra(self):
        return self._tex_ketbra(bra=True)

    def _str_ketbra(self, bra=True):
        coeff, term = self.operands

        if isinstance(coeff, OperatorPlus):
            if bra:
                cs = r"({!s})".format(coeff.adjoint())
            else:
                cs = r"({!s})".format(coeff)
        else:
            if bra:
                cs = "{!s}".format(coeff.adjoint())
            else:
                cs = "{!s}".format(coeff)

        if isinstance(term, KetPlus):
            if bra:
                ct = r"({!s}) ".format(term._str_bra())
            else:
                ct = r" ({!s})".format(term._str_ket())
        else:
            if bra:
                ct = r"{!s} ".format(term._str_bra())
            else:
                ct = r" {!s}".format(term._str_ket())
        if bra:
            return ct + cs
        else:
            return cs + ct

    def _str_ket(self):
        return self._str_ketbra(bra=False)

    def _str_bra(self):
        return self._str_ketbra(bra=True)

    def _expand(self):
        c, t = self.operands
        ct = c.expand()
        et = t.expand()
        if isinstance(et, KetPlus):
            if isinstance(ct, OperatorPlus):
                return sum((cto * eto
                            for eto in et.operands
                            for cto in ct.operands), ZeroKet)
            else:
                return sum((c * eto for eto in et.operands), ZeroKet)
        elif isinstance(ct, OperatorPlus):
            return sum((cto * et for cto in ct.operands), ZeroKet)
        return ct * et

    def _series_expand(self, param, about, order):
        ce = self.coeff.series_expand(param, about, order)
        te = self.term.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class Bra(Operation):
    """The associated dual/adjoint state for any ``Ket`` object ``k`` is given
    by ``Bra(k)``.

    :param Ket k: The state to represent as Bra.
    """

    @property
    def ket(self):
        """The state that is represented as a Bra.

        :rtype: Ket
        """
        return self.operands[0]

    operand = ket

    @cache_attr('_str')
    def __str__(self):
        return self.ket._str_bra()

    @cache_attr('_tex')
    def tex(self):
        return self.ket._tex_bra()

    def adjoint(self):
        """The adjoint of a ``Bra`` is just the original ``Ket`` again.

        :rtype: Ket
        """
        return self.ket
    dag = adjoint

    @property
    def space(self):
        return self.operands[0].space

    def __mul__(self, other):
        if isinstance(other, Ket.scalar_types):
            return Bra.create(self.ket * other.conjugate())
        elif isinstance(other, Operator):
            return Bra.create(other.adjoint() * self.ket)
        elif isinstance(other, Ket):
            return BraKet.create(self.ket, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Ket.scalar_types):
            return Bra.create(self.ket * other.conjugate())
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Bra):
            return Bra.create(self.ket + other.ket)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Bra):
            return Bra.create(self.ket - other.ket)
        return NotImplemented

    def __div__(self, other):
        if isinstance(other, Ket.scalar_types):
            return Bra.create(self.ket/other.conjugate())
        return NotImplemented
    __truediv__ = __div__


class BraKet(Operator, Operation):
    r"""The symbolic inner product between two states, represented as Bra and
    Ket::

        BraKet(b, k)

    In math notation this corresponds to:

    .. math::
        \langle b | k \rangle

    which we define to be linear in the state :math:`k` and anti-linear in
    :math:`b`.

    :param Ket b: The anti-linear state argument.
    :param Ket k: The linear state argument.
    """


    signature = (Ket, Ket), {}
    _rules = []  # see end of module
    _space = TrivialSpace
    _simplifications = [check_kets_same_space, match_replace]

    @property
    def ket(self):
        return self.operands[1]

    @property
    def bra(self):
        return Bra(self.operands[0])

    @property
    def space(self):
        return self.operands[0].space

    def _adjoint(self):
        return BraKet.create(*reversed(self.operands))

    def _expand(self):
        b, k = self.operands
        be, ke = b.expand(), k.expand()
        besummands = be.operands if isinstance(be, KetPlus) else (be,)
        kesummands = ke.operands if isinstance(ke, KetPlus) else (ke,)
        return sum(BraKet.create(bes, kes)
                   for bes in besummands for kes in kesummands)

    @cache_attr('_tex')
    def tex(self):
        if isinstance(self.bra.ket, KetPlus):
            bs = r"\left({}\right)".format(self.bra.tex())
        else:
            bs = self.bra.tex()
        if isinstance(self.ket, KetPlus):
            ks = r"\left({}\right)".format(self.ket.tex())
        else:
            ks = self.ket.tex()
        return bs + ks

    def _series_expand(self, param, about, order):
        be = self.bra.series_expand(param, about, order)
        ke = self.ket.series_expand(param, about, order)
        return tuple(be[k] * ke[n - k]
                     for n in range(order + 1) for k in range(n + 1))


class KetBra(Operator, Operation):
    """A symbolic operator formed by the outer product of two states::

        KetBra(k, b)

    :param Ket k: The first state that defines the range of the operator.
    :param ket b: The second state that defines the Kernel of the operator.
    """

    signature = (Ket, Ket), {}
    _rules = []  # see end of module
    _simplifications = [check_kets_same_space, match_replace]

    @property
    def ket(self):
        return self.operands[0]

    @property
    def bra(self):
        return Bra(self.operands[1])

    @property
    def space(self):
        return self.operands[0].space

    def _adjoint(self):
        return KetBra.create(*reversed(self.operands))

    def _expand(self):
        k, b = self.operands
        be, ke = b.expand(), k.expand()
        kesummands = ke.operands if isinstance(ke, KetPlus) else (ke,)
        besummands = be.operands if isinstance(be, KetPlus) else (be,)
        return sum(KetBra.create(kes, bes)
                   for bes in besummands for kes in kesummands)

    @cache_attr('_tex')
    def tex(self):
        if isinstance(self.bra.ket, KetPlus):
            bs = r"\left({}\right)".format(self.bra.tex())
        else:
            bs = self.bra.tex()
        if isinstance(self.ket, KetPlus):
            ks = r"\left({}\right)".format(self.ket.tex())
        else:
            ks = self.ket.tex()
        return ks + bs

    def _series_expand(self, param, about, order):
        ke = self.ket.series_expand(param, about, order)
        be = self.bra.series_expand(param, about, order)
        return tuple(ke[k] * be[n - k]
                     for n in range(order + 1) for k in range(n + 1))


def act_locally(op, ket):
    ket_on, ket_off = ket.factor_for_space(op.space)
    if ket_off != TrivialKet:
        return (op * ket_on) * ket_off
    raise CannotSimplify()


def act_locally_times_tensor(op, ket):
    local_spaces = op.space.local_factors()
    for spc in local_spaces:
        while spc < ket.space:
            op_on, op_off = op.factor_for_space(spc)
            ket_on, ket_off = ket.factor_for_space(spc)

            if (op_on.space <= ket_on.space and
                    op_off.space <= ket_off.space and ket_off != TrivialKet):
                return (op_on * ket_on) * (op_off * ket_off)
            else:
                spc = op_on.space * ket_on.space
    raise CannotSimplify()


def tensor_decompose_kets(a, b, operation):
    full_space = a.space * b.space
    local_spaces = full_space.local_factors()
    for spc in local_spaces:
        while spc < full_space:
            a_on, a_off = a.factor_for_space(spc)
            b_on, b_off = b.factor_for_space(spc)
            if (a_on.space == b_on.space and a_off.space == b_off.space and
                    a_off != TrivialKet):
                return operation(a_on, b_on) * operation(a_off, b_off)
            else:
                spc = a_on.space * b_on.space
    raise CannotSimplify()





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

A_times = wc("A", head=OperatorTimes)
A_local = wc("A", head=LocalOperator)
B_local = wc("B", head=LocalOperator)

nsym = wc("nsym", head=(int, str, SympyBasic))

Psi = wc("Psi", head=Ket)
Phi = wc("Phi", head=Ket)
Psi_plus = wc("Psi", head=KetPlus)
Psi_local = wc("Psi", head=LocalKet)
Psi_tensor = wc("Psi", head=TensorKet)
Phi_tensor = wc("Phi", head=TensorKet)

ls = wc("ls", head=LocalSpace)

ra = wc("ra", head=(int, str))
rb = wc("rb", head=(int, str))
rc = wc("rc", head=(int, str))
rd = wc("rd", head=(int, str))

ScalarTimesKet._rules += [
    (pattern_head(1, Psi),
        lambda Psi: Psi),
    (pattern_head(0, Psi),
        lambda Psi: ZeroKet),
    (pattern_head(u, ZeroKet),
        lambda u: ZeroKet),
    (pattern_head(u, pattern(ScalarTimesKet, v, Psi)),
        lambda u, v, Psi: (u * v) * Psi)
]

# local_rule = lambda A, B, Psi: OperatorTimes.create(*A) * (B * Psi)

def local_rule(A, B, Psi):
    return OperatorTimes.create(*A) * (B * Psi)

OperatorTimesKet._rules += [
    (pattern_head(IdentityOperator, Psi),
        lambda Psi: Psi),
    (pattern_head(ZeroOperator, Psi),
        lambda Psi: ZeroKet),
    (pattern_head(A, ZeroKet),
        lambda A: ZeroKet),
    (pattern_head(A, pattern(ScalarTimesKet, v, Psi)),
        lambda A, v, Psi:  v *(A* Psi)),

    (pattern_head(pattern(LocalSigma, ls, n, m), pattern(BasisKet, ls, k)),
        lambda ls, n, m, k: BasisKet(ls, n) if m == k else ZeroKet),

    # harmonic oscillator
    (pattern_head(pattern(Create, ls), pattern(BasisKet, ls, n)),
        lambda ls, n: sqrt(n+1) * BasisKet(ls, n + 1)),
    (pattern_head(pattern(Destroy, ls), pattern(BasisKet, ls, n)),
        lambda ls, n: sqrt(n) * BasisKet(ls, n - 1)),
    (pattern_head(pattern(Destroy, ls), pattern(CoherentStateKet, ls, u)),
        lambda ls, u: u * CoherentStateKet(ls, u)),

    # spin
    (pattern_head(pattern(Jplus, ls), pattern(BasisKet, ls, n)),
        lambda ls, n: Jpjmcoeff(ls, n) * BasisKet(ls, n+1)),
    (pattern_head(pattern(Jminus, ls), pattern(BasisKet, ls, n)),
        lambda ls, n: Jmjmcoeff(ls, n) * BasisKet(ls, n-1)),
    (pattern_head(pattern(Jz, ls), pattern(BasisKet, ls, n)),
        lambda ls, n: Jzjmcoeff(ls, n) * BasisKet(ls, n)),

    # # harmonic oscillator
    # (pattern_head(pattern(Create, ls), pattern(BasisKet, ls, nsym)), lambda ls, nsym: sqrt(nsym+1) * BasisKet(ls, nsym + 1)),
    # (pattern_head(pattern(Destroy, ls), pattern(BasisKet, ls, nsym)), lambda ls, nsym: sqrt(nsym) * BasisKet(ls, nsym - 1)),
    # (pattern_head(pattern(Destroy, ls), pattern(CoherentStateKet, ls, u)), lambda ls, u: u * CoherentStateKet(ls, u)),
    #
    # # spin
    # (pattern_head(pattern(Jplus, ls), pattern(BasisKet, ls, nsym)), lambda ls, nsym: Jpjmcoeff(ls, nsym) * BasisKet(ls, nsym+1)),
    # (pattern_head(pattern(Jminus, ls), pattern(BasisKet, ls, nsym)), lambda ls, nsym: Jmjmcoeff(ls, nsym) * BasisKet(ls, nsym-1)),
    # (pattern_head(pattern(Jz, ls), pattern(BasisKet, ls, nsym)), lambda ls, nsym: nsym * BasisKet(ls, nsym)),

    (pattern_head(A_local, Psi_tensor),
        lambda A, Psi: act_locally(A, Psi)),
    (pattern_head(A_times, Psi_tensor),
        lambda A, Psi: act_locally_times_tensor(A, Psi)),
    (pattern_head(A, pattern(OperatorTimesKet, B, Psi)),
        lambda A, B, Psi: (A * B) * Psi if (B * Psi) == OperatorTimesKet(B, Psi) else A * (B * Psi)),
    (pattern_head(pattern(OperatorTimes, A__, B_local), Psi_local),
        local_rule),
    (pattern_head(pattern(ScalarTimesOperator, u, A), Psi),
        lambda u, A, Psi: u * (A * Psi)),
    (pattern_head(pattern(Displace, ls, u), pattern(BasisKet, ls, 0)),
        lambda ls, u: CoherentStateKet(ls, u)),
    (pattern_head(pattern(Displace, ls, u), pattern(CoherentStateKet, ls, v)),
        lambda ls, u, v: (Displace(ls,u) * Displace(ls, v)) * BasisKet(ls, 0)),
    (pattern_head(pattern(Phase, ls, u), pattern(BasisKet, ls, m)),
        lambda ls, u, m: exp(I * u * m) * BasisKet(ls, m)),
    (pattern_head(pattern(Phase, ls, u), pattern(CoherentStateKet, ls, v)),
        lambda ls, u, v: CoherentStateKet(ls, v * exp(I * u))),
]

KetPlus._binary_rules += [
    (pattern_head(pattern(ScalarTimesKet, u, Psi),
                  pattern(ScalarTimesKet, v, Psi)),
        lambda u, v, Psi: (u + v) * Psi),
    (pattern_head(pattern(ScalarTimesKet, u, Psi), Psi),
        lambda u, Psi: (u + 1) * Psi),
    (pattern_head(Psi, pattern(ScalarTimesOperator, v, Psi)),
        lambda v, Psi: (1 + v) * Psi),
    (pattern_head(Psi, Psi),
        lambda Psi: 2 * Psi),
]

TensorKet._binary_rules += [
    (pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
        lambda u, Psi, Phi: u * (Psi * Phi)),
    (pattern_head(Psi, pattern(ScalarTimesKet, u, Phi)),
        lambda Psi, u, Phi: u * (Psi * Phi)),
]

BraKet._rules += [
    (pattern_head(Phi, ZeroKet),
        lambda Phi: ZeroOperator),
    (pattern_head(ZeroKet, Phi),
        lambda Phi: ZeroOperator),
    (pattern_head(pattern(BasisKet, ls, m), pattern(BasisKet, ls, n)),
        lambda ls, m, n: IdentityOperator if m == n else ZeroOperator),
    (pattern_head(pattern(BasisKet, ls, nsym), pattern(BasisKet, ls, nsym)),
        lambda ls, nsym: IdentityOperator),
    (pattern_head(Psi_tensor, Phi_tensor),
        lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, BraKet.create)),
    (pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
        lambda u, Psi, Phi: u.conjugate() * (Psi.adjoint() * Phi)),
    (pattern_head(pattern(OperatorTimesKet, A, Psi), Phi),
        lambda A, Psi, Phi: (Psi.adjoint() * (A.dag() * Phi))),
    (pattern_head(Psi, pattern(ScalarTimesKet, u,Phi)),
        lambda Psi, u, Phi: u * (Psi.adjoint() * Phi)),
]

KetBra._rules += [
    (pattern_head(pattern(BasisKet, ls, m), pattern(BasisKet, ls, n)),
        lambda ls, m, n: LocalSigma(ls, m, n)),
    (pattern_head(pattern(CoherentStateKet, ls, u), Phi),
        lambda ls, u, Phi: Displace(ls, u) * (BasisKet(ls, 0) *  Phi.adjoint())),
    (pattern_head(Phi, pattern(CoherentStateKet, ls, u)),
        lambda ls, u, Phi: (Phi * BasisKet(ls, 0).adjoint()) * Displace(ls, -u)),
    (pattern_head(Psi_tensor,Phi_tensor),
        lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, KetBra.create)),
    (pattern_head(pattern(OperatorTimesKet, A, Psi), Phi),
        lambda A, Psi, Phi: A * (Psi * Phi.adjoint())),
    (pattern_head(Psi, pattern(OperatorTimesKet, A,Phi)),
        lambda Psi, A, Phi: (Psi * Phi.adjoint()) * A.adjoint()),
    (pattern_head(pattern(ScalarTimesKet, u, Psi), Phi),
        lambda u, Psi, Phi: u * (Psi * Phi.adjoint())),
    (pattern_head(Psi, pattern(ScalarTimesKet, u,Phi)),
        lambda Psi, u, Phi: u.conjugate() * (Psi * Phi.adjoint())),
]
