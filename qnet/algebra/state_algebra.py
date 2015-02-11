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

For more details see :ref:`state_algebra`.

"""


from qnet.algebra.operator_algebra import *

@six.add_metaclass(ABCMeta)
class Ket(object):
    """
    Basic Ket algebra class to represent Hilbert Space states
    """

    scalar_types = Operator.scalar_types

    @property
    def space(self):
        """
        The associated Hilbert space.

        :rtype: HilbertSpace
        """
        return self._space

    @abstractproperty
    def _space(self):
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """
        The adjoint of a Ket state, i.e., the corresponding Bra.

        :rtype: Bra
        """
        return Bra(self)

    dag = adjoint


    def to_qutip(self):
        """
        Create a numerical representation of the ket as a QuTiP object.
        Note that all symbolic scalar parameters need to be replaced by numerical values before calling this method.

        :return: The numerical representation of the operator.
        :rtype: qutip.Qobj
        """
        return self._to_qutip()

    @abstractmethod
    def _to_qutip(self):
        raise NotImplementedError(str(self.__class__))

    def expand(self):
        """
        Expand out distributively all products of sums. Note that this does not expand out sums of scalar coefficients.

        :return: A fully expanded sum of states.
        :rtype: Ket
        """
        return self._expand()

    @abstractmethod
    def _expand(self):
        raise NotImplementedError(self.__class__.__name__)

#    def series_expand(self, param, about, order):
#        """
#        Expand the state expression as a truncated power series in a scalar parameter.
#
#        :param param: Expansion parameter.
#        :type param: sympy.core.symbol.Symbol
#        :param about: Point about which to expand.
#        :type about:  Any one of Operator.scalar_types
#        :param order: Maximum order of expansion.
#        :type order: int >= 0
#        """
#        return self._series_expand(param, about, order)
#
#    @abstractmethod
#    def _series_expand(self, param, about, order):
#        raise NotImplementedError(self.__class__.__name__)

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

    def __str__(self):
        return self._str_ket()

    def _tex(self):
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



@check_signature
class KetSymbol(Ket, Operation):
    """
    Ket symbol class, parametrized by an identifier string and an associated Hilbert space.

        ``KetSymbol(name, hs)``

    :param name: Symbol identifier
    :type name: str
    :param hs: Associated Hilbert space.
    :type hs: HilbertSpace

    """


    signature = str, (HilbertSpace, str, int)

    def __init__(self, name, hs):
        if isinstance(hs, (str, int)):
            hs = local_space(hs)
        super(KetSymbol, self).__init__(name, hs)


    def _str_ket(self):
        return "|" + self.operands[0] + ">"

    def _str_bra(self):
        return "<" + self.operands[0] + "|"


    def _tex_ket(self):
        return "\left| "+identifier_to_tex(self.operands[0]) + r"\right\rangle_{" + self.space.tex() + "}"

    def _tex_bra(self):
        return "\left\langle "+identifier_to_tex(self.operands[0]) + r"\right|_{" + self.space.tex() + "}"


    def _to_qutip(self):
        raise AlgebraError("Cannot convert operator symbol to  numeric representation. Substitute first.")

    @property
    def _space(self):
        return self.operands[1]

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)

    def _all_symbols(self):
        return {self}


@singleton
class KetZero(Ket, Expression):
    """
    KetZero constant (singleton) object for the null-state.
    """

    @property
    def _space(self):
        return FullSpace

    def _to_qutip(self):
        raise ValueError("Can't represent the ZeroKet numerically.")

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

    def _all_symbols(self):
        return set(())

@singleton
class TrivialKet(Ket, Expression):
    """
    TrivialKet constant (singleton) object.
    This is the neutral element under the state tensor-product.
    """

    @property
    def _space(self):
        return TrivialSpace

    def _adjoint(self):
        return Bra(TrivialKet)

    def _to_qutip(self):
        raise AlgebraError("Cannot convert TrivialKet to numeric representation.")

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)


    def _tex_ket(self):
        return r"\left| {\rm id}\right \rangle"

    def _tex_bra(self):
        return r"\left\langle {\rm id}\right |"


    def __eq__(self, other):
        return self is other or other == 1

    def _str_ket(self):
        return "|id>"

    def _str_bra(self):
        return "<id|"

    def _all_symbols(self):
        return set(())

class LocalKet(Ket, Operation):
    """
    Base class for atomic (non-composite) ket states of single degrees of freedom.
    """

    def __init__(self, hs, *args):
        if isinstance(hs, (str, int)):
            hs = local_space(hs)
        super(LocalKet, self).__init__(hs, *args)

    @property
    def _space(self):
        return self.operands[0]

    def _to_qutip(self):
        raise NotImplementedError(self.__class__.__name__)

    def _expand(self):
        return self

    def _series_expand(self, param, about, order):
        return (self,) + (0,)*(order - 1)


    def _all_symbols(self):
        return {}


@check_signature
class BasisKet(LocalKet):
    """
    Local basis state, labeled by an integer or a string.

    Instantiate as::

        BasisKet(hs, rep)

    :param LocalSpace hs: The local Hilbert space degree of freedom.
    :param (str or int) rep: The basis state label.
    """
    signature = (LocalSpace, str, int), (str, int)

    def _tex_ket(self):
        return r"\left| {!s} \right\rangle_{{{}}}".format(self.operands[1], self.space.tex())

    def _tex_bra(self):
        return r"\left\langle {!s} \right|_{{{}}}".format(self.operands[1], self.space.tex())

    def _str_ket(self):
        return r"|{!s}>_{!s}".format(self.operands[1], self.space)

    def _str_bra(self):
        return r"<{!s}|_{!s}".format(self.operands[1], self.space)

    def _to_qutip(self):
        return qutip.basis(self.space.dimension, self.space.basis.index(self.operands[1]))



@check_signature
class CoherentStateKet(LocalKet):
    """
    Local coherent state, labeled by a scalar amplitude.

    Instantiate as::

        CoherentStateKet(hs, amp)

    :param LocalSpace hs: The local Hilbert space degree of freedom.
    :param Ket.scalar_types amp: The coherent displacement amplitude.
    """

    signature = (LocalSpace, str, int), Ket.scalar_types

    def _tex_ket(self):
        return r"\left| D\left({!s}\right) \right\rangle_{{{}}}".format(tex(self.operands[1]), self.space.tex())

    def _tex_bra(self):
        return r"\left\langle D\left({!s}\right) \right|_{{{}}}".format(tex(self.operands[1]), self.space.tex())

    def _str_ket(self):
        return r"|D({!s})>_{!s}".format(self.operands[1], self.space)

    def _str_bra(self):
        return r"<D({!s})|_{!s}".format(self.operands[1], self.space)

    def _to_qutip(self):
        return qutip.coherent(self.space.dimension, complex(self.operands[1]))

    def _series_expand(self, param, about, order):
        return (self,) + (0,) * (order - 1)

    def _substitute(self, var_map):
        hs, amp = self.operands
        if isinstance(amp, SympyBasic):
            svar_map = {k:v for k,v in var_map.items() if not isinstance(k,Expression)}
            ampc = amp.subs(svar_map)
        else:
            ampc = substitute(amp, var_map)
            
        return CoherentStateKet(hs, ampc)
            


class UnequalSpaces(AlgebraError):
    pass

def _check_same_space_mtd(dcls, clsmtd, cls, *ops):
    """
    Check that all operands are from the same Hilbert space.
    """
    if not len({o.space for o in ops}) == 1:
        raise UnequalSpaces(str(ops))
    return clsmtd(cls, *ops)

check_same_space = preprocess_create_with(_check_same_space_mtd)


@assoc
@orderby
@filter_neutral
@check_same_space
@match_replace_binary
@filter_neutral
@check_signature_assoc
class KetPlus(Ket, Operation):
    """
    A sum of Ket states.

    Instantiate as::

        KetPlus(*summands)

    :param summands: State summands.
    :type summands: Ket
    """
    signature = Ket,
    neutral_element = KetZero
    _binary_rules = []


    @classmethod
    def order_key(cls, a):
        if isinstance(a, ScalarTimesKet):
            return Operation.order_key(a.term), a.coeff
        return Operation.order_key(a), 1

    def _to_qutip(self):
        return sum((op.to_qutip() for op in self.operands), 0)

    def _expand(self):
        return sum((o.expand() for o in self.operands), KetZero)

    def _series_expand(self, param, about, order):
        res = sum((o.series_expand(param, about, order) for o in self.operands), KetZero)
        return res

    def _tex_ketbra(self, bra = False):
        ret = self.operands[0].tex()

        for o in self.operands[1:]:
            if isinstance(o, ScalarTimesKet) and ScalarTimesKet.has_minus_prefactor(o.coeff):
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
        return self._tex_ketbra(bra = False)

    def _tex_bra(self):
        return self._tex_ketbra(bra = True)

    def _str_ketbra(self, bra = False):
        ret = str(self.operands[0])

        for o in self.operands[1:]:
            if isinstance(o, ScalarTimesKet) and ScalarTimesKet.has_minus_prefactor(o.coeff):
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
    def _space(self):
        return self.operands[0].space

class OverlappingSpaces(AlgebraError):
    pass

@assoc
@orderby
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_assoc
class TensorKet(Ket, Operation):
    """
    A tensor product of kets each belonging to different degrees of freedom.
    Instantiate as::

        TensorKet(*factors)

    :param factors: Ket factors.
    :type factors: Ket
    """
    signature = Ket,
    _binary_rules = []
    neutral_element = TrivialKet

    order_key = OperatorTimes.order_key

    @classmethod
    def create(cls, *ops):
        if any(o == KetZero for o in ops):
            return KetZero
        spc = TrivialSpace
        for o in ops:
            if o.space & spc > TrivialSpace:
                raise OverlappingSpaces(str(ops))
            spc *= o.space

        return super(TensorKet, cls).create(*ops)

    def factor_for_space(self, space):
        if not space <= self.space:
            raise SpaceTooLargeError(str((self, space)))
        if space == self.space:
            return self
        if space is TrivialSpace:
            on_ops = [o for o in self.operands if o.space is TrivialSpace]
            off_ops = [o for o in self.operands if o.space > TrivialSpace]
        else:
            on_ops = [o for o in self.operands if o.space & space > TrivialSpace]
            off_ops = [o for o in self.operands if o.space & space is TrivialSpace]
        return TensorKet.create(*on_ops), TensorKet.create(*off_ops)



    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)

    def _to_qutip(self):

        # if any factor acts non-locally, we need to expand distributively.
        if any(len(op.space) > 1 for op in self.operands):
            se = self.expand()
            if se == self:
                raise ValueError("Cannot represent as QuTiP object: {!s}".format(self))
            return se.to_qutip()

        return qutip.tensor(*[o.to_qutip() for o in self.operands])

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [eo.operands if isinstance(eo, KetPlus) else (eo,) for eo in eops]
        # iterate over a cartesian product of all factor summands, form product of each tuple and sum over result
        return sum((TensorKet.create(*combo) for combo in cartesian_product(*eopssummands)), KetZero)

    def _tex_ketbra(self, bra = False):
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
        return self._tex_ketbra(bra = False)

    def _tex_bra(self):
        return self._tex_ketbra(bra = True)

    def _str_ketbra(self, bra = False):
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
        return self._str_ketbra(bra = False)

    def _str_bra(self):
        return self._str_ketbra(bra = True)

@match_replace
@check_signature
class ScalarTimesKet(Ket, Operation):
    """
    Multiply a Ket by a scalar coefficient.

    Instantiate as::
        ScalarTimesKet(coefficient, term)

    :param coefficient: Scalar coefficient.
    :type coefficient: Operator.scalar_types
    :param term: The ket that is multiplied.
    :type term: Ket
    """
    signature = Ket.scalar_types, Ket
    _rules = []


    @property
    def _space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]


    def _tex_ketbra(self, bra = False):
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
        return self._tex_ketbra(bra = False)

    def _tex_bra(self):
        return self._tex_ketbra(bra = True)



    def _str_ketbra(self, bra = False):
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
        return self._str_ketbra(bra = False)

    def _str_bra(self):
        return self._str_ketbra(bra = True)


    def _to_qutip(self):
        return complex(self.coeff) * self.term.to_qutip()

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, KetPlus):
            return sum((c * eto for eto in et.operands), KetZero)
        return c * et

    def _series_expand(self, param, about, order):
        ceg = sympy_series(self.coeff, x=param, x0=about, n=None)
        ce = tuple(ceo for ceo, k in zip(ceg, range(order + 1)))
        te = self.term.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k] for n in range(order + 1) for k in range(n + 1))

    def _substitute(self, var_map):
        st = self.term.substitute(var_map)
        if isinstance(self.coeff, SympyBasic):
            svar_map = {k:v for k,v in var_map.items() if not isinstance(k, Expression)}
            sc = self.coeff.subs(svar_map)
        else:
            sc = substitute(self.coeff, var_map)
        return sc * st


class SpaceTooLargeError(AlgebraError):
    pass


@match_replace
@check_signature
class OperatorTimesKet(Ket, Operation):
    """
    Multiply an operator by a scalar coefficient.

    Instantiate as::

        OperatorTimesKet(op, ket)

    :param Operator op: The multiplying operator.
    :param Ket ket: The ket that is multiplied.
    """
    signature = Operator, Ket
    _rules = []

    @classmethod
    def create(cls, op, ket):
        if not op.space <= ket.space:
            raise SpaceTooLargeError(str(op) + " " + str(ket))
        return super(OperatorTimesKet, cls).create(op, ket)


    @property
    def _space(self):
        return self.operands[1].space

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]


    def _tex_ketbra(self, bra = False):
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
        return self._tex_ketbra(bra = False)

    def _tex_bra(self):
        return self._tex_ketbra(bra = True)



    def _str_ketbra(self, bra = True):
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
        return self._str_ketbra(bra = False)

    def _str_bra(self):
        return self._str_ketbra(bra = True)



    def _to_qutip(self):
        return self.coeff.to_qutip(self.space) * self.term.to_qutip()

    def _expand(self):
        c, t = self.operands
        ct = c.expand()
        et = t.expand()
        if isinstance(et, KetPlus):
            if isinstance(ct, OperatorPlus):
                return sum((cto * eto for eto in et.operands for cto in ct.operands), KetZero)
            else:
                return sum((c * eto for eto in et.operands), KetZero)
        elif isinstance(ct, OperatorPlus):
            return sum((cto * et for cto in ct.operands), KetZero)
        return ct * et

    def _series_expand(self, param, about, order):
        ce = self.coeff.series_expand(param, about, order)
        te = self.term.series_expand(param, about, order)

        return tuple(ce[k] * te[n - k] for n in range(order + 1) for k in range(n + 1))

@check_signature
class Bra(Operation):
    """
    The associated dual/adjoint state for any ``Ket`` object ``k`` is given by ``Bra(k)``.

    :param Ket k: The state to represent as Bra.
    """

    signature = Ket,

    @property
    def ket(self):
        """
        The state that is represented as a Bra.

        :rtype: Ket
        """
        return self.operands[0]

    def __str__(self):
        return self.ket._str_bra()

    def _tex(self):
        return self.ket._tex_bra()

    def adjoint(self):
        """
        The adjoint of a ``Bra`` is just the original ``Ket`` again.

        :rtype: Ket
        """
        return self.ket
    dag = adjoint

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

@check_same_space
@match_replace
@check_signature
class BraKet(Operator, Operation):
    r"""
    The symbolic inner product between two states, represented as Bra and Ket::

        BraKet(b, k)

    In math notation this corresponds to:

    .. math::
        \langle b | k \rangle

    which we define to be linear in the state :math:`k` and anti-linear in :math:`b`.

    :param Ket b: The anti-linear state argument.
    :param Ket k: The linear state argument.
    """


    signature = Ket, Ket
    _rules = []
    _space = TrivialSpace

    @property
    def ket(self):
        return self.operands[1]

    @property
    def bra(self):
        return Bra(self.operands[0])

    def _adjoint(self):
        return BraKet.create(*reversed(self.operands))

    def _expand(self):
        b, k = self.operands
        be, ke = b.expand(), k.expand()
        besummands = be.operands if isinstance(be, KetPlus) else (be,)
        kesummands = ke.operands if isinstance(ke, KetPlus) else (ke,)
        return sum(BraKet.create(bes, kes) for bes in besummands for kes in kesummands)

    def _tex(self):
        if isinstance(self.bra.ket, KetPlus):
            bs = r"\left({}\right)".format(self.bra.tex())
        else:
            bs = self.bra.tex()
        if isinstance(self.ket, KetPlus):
            ks = r"\left({}\right)".format(self.ket.tex())
        else:
            ks = self.ket.tex()
        return bs + ks



@check_same_space
@match_replace
@check_signature
class KetBra(Operator, Operation):
    """
    A symbolic operator formed by the outer product of two states::

        KetBra(k, b)

    :param Ket k: The first state that defines the range of the operator.
    :param ket b: The second state that defines the Kernel of the operator.
    """

    signature = Ket, Ket
    _rules = []

    @property
    def ket(self):
        return self.operands[0]

    @property
    def bra(self):
        return Bra(self.operands[1])

    @property
    def _space(self):
        return self.operands[0].space

    def _adjoint(self):
        return KetBra.create(*reversed(self.operands))

    def _expand(self):
        k, b = self.operands
        be, ke = b.expand(), k.expand()
        kesummands = ke.operands if isinstance(ke, KetPlus) else (ke,)
        besummands = be.operands if isinstance(be, KetPlus) else (be,)
        return sum(KetBra.create(kes, bes) for bes in besummands for kes in kesummands)

    def _tex(self):
        if isinstance(self.bra.ket, KetPlus):
            bs = r"\left({}\right)".format(self.bra.tex())
        else:
            bs = self.bra.tex()
        if isinstance(self.ket, KetPlus):
            ks = r"\left({}\right)".format(self.ket.tex())
        else:
            ks = self.ket.tex()
        return ks + bs





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

            if op_on.space <= ket_on.space and op_off.space <= ket_off.space and ket_off != TrivialKet:
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

                if a_on.space == b_on.space and a_off.space == b_off.space and a_off != TrivialKet:
                    return operation(a_on, b_on) * operation(a_off, b_off)
                else:
                    spc = a_on.space * b_on.space
    raise CannotSimplify()





## Expression rewriting _rules
#u = wc("u", head=Operator.scalar_types)
#v = wc("v", head=Operator.scalar_types)
#
#n = wc("n", head=(int, str))
#m = wc("m", head=(int, str))
k = wc("k", head=(int, str))

Psi = wc("Psi", head=Ket)
Phi = wc("Phi", head=Ket)
Psi_plus = wc("Psi", head=KetPlus)
Psi_local = wc("Psi", head=LocalKet)
Psi_tensor = wc("Psi", head=TensorKet)
Phi_tensor = wc("Phi", head=TensorKet)

#ls = wc("ls", head=LocalSpace)
#h1 = wc("h1", head = HilbertSpace)
#h2 = wc("h2", head = HilbertSpace)
#H_ProductSpace = wc("H", head = ProductSpace)

#ra = wc("ra", head=(int, str))
#rb = wc("rb", head=(int, str))
#rc = wc("rc", head=(int, str))
#rd = wc("rd", head=(int, str))

ScalarTimesKet._rules += [
    ((1, Psi), lambda Psi: Psi),
    ((0, Psi), lambda Psi: KetZero),
    ((u, KetZero), lambda u: KetZero),
    ((u, ScalarTimesKet(v, Psi)), lambda u, v, Psi: (u * v) * Psi)
]

OperatorTimesKet._rules += [
    ((IdentityOperator, Psi), lambda Psi: Psi),
    ((ZeroOperator, Psi), lambda Psi: KetZero),
    ((A, KetZero), lambda u: KetZero),
    ((A, ScalarTimesKet(v, Psi)), lambda A, v, Psi:  v *(A* Psi)),
    ((LocalSigma(ls, n, m), BasisKet(ls, k)), lambda ls, n, m, k: BasisKet(ls, n) if m == k else KetZero),
    ((Create(ls), BasisKet(ls, n)), lambda ls, n: sqrt(n+1) * BasisKet(ls, n + 1)),
    ((Destroy(ls), BasisKet(ls, n)), lambda ls, n: sqrt(n) * BasisKet(ls, n - 1)),
    ((Destroy(ls), CoherentStateKet(ls, u)), lambda ls, u: u * CoherentStateKet(ls, u)),
    ((A_local, Psi_tensor), lambda A, Psi: act_locally(A, Psi)),
    ((A_times, Psi_tensor), lambda A, Psi: act_locally_times_tensor(A, Psi)),
    ((A, OperatorTimesKet(B, Psi)), lambda A, B, Psi: (A * B) * Psi if (B * Psi) == OperatorTimesKet(B, Psi) else A * (B * Psi)),
    ((OperatorTimes(A__, B_local), Psi_local), lambda A, B, Psi: OperatorTimes.create(*A) * (B * Psi)),
    ((ScalarTimesOperator(u, A), Psi), lambda u, A, Psi: u * (A * Psi)),
    ((Displace(ls, u), BasisKet(ls, 0)), lambda ls, u: CoherentStateKet(ls, u)),
    ((Displace(ls, u), CoherentStateKet(ls, v)), lambda ls, u, v: (Displace(ls,u) * Displace(ls, v)) * BasisKet(ls, 0)),
    ((Phase(ls, u), BasisKet(ls, m)), lambda ls, u, m: exp(I * u * m) * BasisKet(ls, m)),
    ((Phase(ls, u), CoherentStateKet(ls, v)), lambda ls, u, v: CoherentStateKet(ls, v * exp(I * u))),
]

KetPlus._binary_rules += [
    ((ScalarTimesKet(u, Psi), ScalarTimesKet(v, Psi)), lambda u, v, Psi: (u + v) * Psi),
    ((ScalarTimesKet(u, Psi), Psi), lambda u, Psi: (u + 1) * Psi),
    ((Psi, ScalarTimesOperator(v, Psi)), lambda v, Psi: (1 + v) * Psi),
    ((Psi, Psi), lambda Psi: 2 * Psi),
]

TensorKet._binary_rules += [
    ((ScalarTimesKet(u, Psi), Phi), lambda u, Psi, Phi: u * (Psi * Phi)),
    ((Psi, ScalarTimesKet(u, Phi)), lambda Psi, u, Phi: u * (Psi * Phi)),
]

BraKet._rules += [
    ((BasisKet(ls, m), BasisKet(ls, n)), lambda ls, m, n: IdentityOperator if m == n else ZeroOperator),
    ((Psi_tensor, Phi_tensor), lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, BraKet.create)),
    ((ScalarTimesKet(u, Psi), Phi), lambda u, Psi, Phi: u.conjugate() * (Psi.adjoint() * Phi)),
    ((Psi, ScalarTimesKet(u,Phi)), lambda Psi, u, Phi: u * (Psi.adjoint() * Phi)),
]

KetBra._rules += [
    ((BasisKet(ls, m),BasisKet(ls, n)), lambda ls, m, n: LocalSigma(ls, m, n)),
    ((CoherentStateKet(ls, u),Phi), lambda ls, u, Phi: Displace(ls, u) * (BasisKet(ls, 0) *  Phi.adjoint())),
    ((Phi, CoherentStateKet(ls, u)), lambda ls, u, Phi: (Phi * BasisKet(ls, 0).adjoint()) * Displace(ls, -u)),
    ((Psi_tensor,Phi_tensor), lambda Psi, Phi: tensor_decompose_kets(Psi, Phi, KetBra.create)),
    ((OperatorTimesKet(A,Psi),Phi), lambda A, Psi, Phi: A * (Psi * Phi.adjoint())),
    ((Psi,OperatorTimesKet(A,Phi)), lambda Psi, A, Phi: (Psi * Phi.adjoint()) * A.adjoint()),
    ((ScalarTimesKet(u,Psi), Phi), lambda u, Psi, Phi: u * (Psi * Phi.adjoint())),
    ((Psi, ScalarTimesKet(u,Phi)), lambda Psi, u, Phi: u.conjugate() * (Psi * Phi.adjoint())),
]
