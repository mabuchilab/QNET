"""Implementation of the scalar (quantum) algebra"""
from abc import ABCMeta
from collections import OrderedDict
from itertools import product as cartesian_product

import numpy
import sympy
from sympy.concrete.delta import _simplify_delta
from numpy import complex128, float64, int64

from .abstract_quantum_algebra import (
    QuantumExpression, QuantumIndexedSum, QuantumOperation, QuantumPlus,
    QuantumTimes, QuantumDerivative)
from .algebraic_properties import (
    assoc, assoc_indexed, convert_to_scalars, filter_neutral,
    indexed_sum_over_const, indexed_sum_over_kronecker, match_replace,
    match_replace_binary, orderby, collect_scalar_summands)
from .hilbert_space_algebra import TrivialSpace
from ...utils.singleton import Singleton, singleton_object
from ...utils.ordering import KeyTuple
from ...utils.indices import SymbolicLabelBase, IdxSym

__all__ = [
    'Scalar', 'ScalarValue', 'ScalarExpression', 'Zero', 'One', 'ScalarPlus',
    'ScalarTimes', 'ScalarIndexedSum', 'ScalarPower', 'ScalarDerivative',
    'sqrt', 'KroneckerDelta']

__private__ = ['is_scalar']


class Scalar(QuantumExpression, metaclass=ABCMeta):
    """Base class for Scalars"""

    #: types that may be wrapped by :class:`ScalarValue`
    _val_types = (
        int, float, complex, sympy.Basic, int64, complex128, float64)

    #: values that cannot be wrapped by :class:`ScalarValue`
    _invalid = {sympy.oo, sympy.zoo, numpy.nan, numpy.inf}

    @property
    def space(self):
        """:obj:`.TrivialSpace`, by definition"""
        return TrivialSpace

    def conjugate(self):
        """Complex conjugate"""
        return self._adjoint()

    @property
    def real(self):
        """Real part"""
        return (self.conjugate() + self) / 2

    @property
    def imag(self):
        """Imaginary part"""
        return (self.conjugate() - self) * (sympy.I / 2)

    def __add__(self, other):
        if other == 0:
            return self
        return super().__add__(other)

    def __sub__(self, other):
        if other == 0:
            return self
        return super().__sub__(other)

    def __mul__(self, other):
        if other == 1:
            return self
        elif other == 0:
            return Zero
        return super().__mul__(other)

    def __floordiv__(self, other):
        if other == 1:
            return self // 1  # Note: 3.5 // 1 == 3.0 != 3.5; -> NOT self
        elif other == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        try:
            # noinspection PyUnresolvedReferences
            return super().__floordiv__(other)
        except AttributeError:
            return NotImplemented

    def __truediv__(self, other):
        if other == 1:
            return self
        elif other == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        elif other == self:
            return One
        if isinstance(other, ScalarValue):
            other = other.val
        if isinstance(other, (float, complex, complex128, float64)):
            return (ScalarValue(1/other)) * self
        elif isinstance(other, (int, sympy.Basic, int64)):
            return (ScalarValue(sympy.sympify(1)/other)) * self
        return super().__truediv__(other)

    def __mod__(self, other):
        if other == 1:
            return Zero
        elif other == 0:
            raise ZeroDivisionError("integer division or modulo by zero")
        try:
            # noinspection PyUnresolvedReferences
            return super().__mod__(other)
        except AttributeError:
            return NotImplemented

    #  __pow__(self, other) is fully implemented in QuantumExpression

    def __radd__(self, other):
        if other == 0:
            return self
        return super().__radd__(other)

    def __rsub__(self, other):
        if other == 0:
            return -self
        return super().__rsub__(other)

    def __rmul__(self, other):
        if other == 1:
            return self
        elif other == 0:
            return Zero
        return super().__rmul__(other)

    def __rfloordiv__(self, other):
        if other == 0:
            return Zero
        try:
            # noinspection PyUnresolvedReferences
            return super().__rfloordiv__(other)
        except AttributeError:
            return NotImplemented

    def __rtruediv__(self, other):
        if other == 0:
            if self != 0:
                return Zero
        elif other == 1:
            return ScalarPower.create(self, -1)
        elif is_scalar(other):
            return other * ScalarPower.create(self, -1)
        try:
            # noinspection PyUnresolvedReferences
            return super().__rtruediv__(other)
        except AttributeError:
            return NotImplemented

    def __rmod__(self, other):
        if other == 0:
            return Zero
        try:
            # noinspection PyUnresolvedReferences
            return super().__rmod__(other)
        except AttributeError:
            return NotImplemented

    def __rpow__(self, other):
        if other == 0:
            return Zero
        elif other == 1:
            return One
        try:
            # noinspection PyUnresolvedReferences
            return super().__rpow__(other)
        except AttributeError:
            return NotImplemented


class ScalarValue(Scalar):
    """Wrapper around a numeric or symbolic value

    The wrapped value may be of any of the following types::

        >>> for t in ScalarValue._val_types:
        ...     print(t)
        <class 'int'>
        <class 'float'>
        <class 'complex'>
        <class 'sympy.core.basic.Basic'>
        <class 'numpy.int64'>
        <class 'numpy.complex128'>
        <class 'numpy.float64'>

    A :class:`ScalarValue` behaves exactly like its wrapped value in all
    algebraic contexts::

        >>> 5 * ScalarValue.create(2)
        10

    Any unknown attributes or methods will be forwarded to the wrapped value
    to ensure complete "duck-typing"::

        >>> alpha = ScalarValue(sympy.symbols('alpha', positive=True))
        >>> alpha.is_positive   # same as alpha.val.is_positive
        True
        >>> ScalarValue(5).is_positive
        Traceback (most recent call last):
          ...
        AttributeError: 'int' object has no attribute 'is_positive'
    """

    @classmethod
    def create(cls, val):
        """Instatiate the :class:`ScalarValue` while recognizing :class:`Zero`
        and :class:`One`.

        :class:`Scalar` instances as `val` (including
        :class:`ScalarExpression` instances) are left unchanged. This makes
        :meth:`ScalarValue.create` a safe method for converting unknown objects
        to :class:`Scalar`.
        """
        if val in cls._invalid:
            raise ValueError("Invalid value %r" % val)
        if val == 0:
            return Zero
        elif val == 1:
            return One
        elif isinstance(val, Scalar):
            return val
        else:
            # We instantiate ScalarValue directly to avoid the overhead of
            # super().create(). Thus, there is no caching for scalars (which is
            # probably a good thing)
            return cls(val)

    def __init__(self, val):
        self._val = val
        if not isinstance(val, self._val_types):
            raise TypeError(
                "val must be one of " +
                ", ".join(["%s" % t for t in self._val_types]))
        super().__init__(val)

    def __getattr__(self, name):
        return getattr(self.val, name)

    def _diff(self, sym):
        if isinstance(self.val, sympy.Basic):
            return ScalarValue.create(sympy.diff(self.val, sym))
        else:
            return Zero

    def _simplify_scalar(self, func):
        if isinstance(self.val, sympy.Basic):
            return self.__class__.create(func(self.val))
        else:
            return self

    @property
    def val(self):
        """The wrapped scalar value"""
        return self._val

    @property
    def args(self):
        """Tuple containing the wrapped scalar value as its only element"""
        return (self._val,)

    def _series_expand(self, param, about, order):
        if isinstance(self.val, sympy.Basic):
            if about != 0:
                c = self.val.subs({param: about + param})
            else:
                c = self.val
            series = sympy.series(c, x=param, x0=0, n=None)
            res = []
            next_order = 0
            for term in series:
                c, o = term.as_coeff_exponent(param)
                if o < 0 or o.is_noninteger:
                    raise ValueError(
                        "%s is singular at expansion point %s=%s."
                        % (self, param, about))
                if o > order:
                    break
                res.extend([0] * (o - next_order))
                res.append(c)
                next_order = o + 1
            res.extend([0] * (order + 1 - next_order))
            return tuple([ScalarValue.create(c) for c in res])
        else:
            return tuple([self, ] + [Zero] * order)

    @property
    def real(self):
        """Real part"""
        if hasattr(self.val, 'real'):
            return self.val.real
        else:
            # SymPy
            return self.val.as_real_imag()[0]

    @property
    def imag(self):
        """Imaginary part"""
        if hasattr(self.val, 'imag'):
            return self.val.imag
        else:
            # SymPy
            return self.val.as_real_imag()[1]

    def _adjoint(self):
        return self.__class__(self.val.conjugate())

    def __eq__(self, other):
        if isinstance(other, ScalarValue):
            return self.val == other.val
        else:
            return self.val == other

    def __lt__(self, other):
        if isinstance(other, ScalarValue):
            return self.val < other.val
        else:
            return self.val < other

    def __le__(self, other):
        if isinstance(other, ScalarValue):
            return self.val <= other.val
        else:
            return self.val <= other

    def __gt__(self, other):
        if isinstance(other, ScalarValue):
            return self.val > other.val
        else:
            return self.val > other

    def __ge__(self, other):
        if isinstance(other, ScalarValue):
            return self.val >= other.val
        else:
            return self.val >= other

    def __hash__(self):
        return hash(self.val)

    def __neg__(self):
        return self.create(-self.val)

    def __abs__(self):
        return self.create(abs(self.val))

    def __add__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val + other.val)
        elif isinstance(other, self._val_types):
            return self.create(self.val + other)
        elif other == 1:
            return self.create(self.val + 1)
        else:
            return super().__add__(other)  # other == 0

    def __sub__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val - other.val)
        elif isinstance(other, self._val_types):
            return self.create(self.val - other)
        elif other == 1:
            return self.create(self.val - 1)
        else:
            return super().__sub__(other)  # other == 0

    def __mul__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val * other.val)
        elif isinstance(other, self._val_types):
            return self.create(self.val * other)
        else:
            return super().__mul__(other)  # other == 0, 1

    def __floordiv__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val // other.val)
        elif isinstance(other, self._val_types):
            return self.create(self.val // other)
        else:
            return super().__floordiv__(other)  # other == 0, 1

    def __truediv__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val / other.val)
        elif isinstance(other, self._val_types):
            try:
                return self.create(self.val / other)
            except ValueError:
                # sympy may produce 'infinity', which `create` catches as a
                # ValueError
                raise ZeroDivisionError("integer division or modulo by zero")
        else:
            return super().__truediv__(other)  # other == 0, 1

    def __mod__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val % other.val)
        elif isinstance(other, self._val_types):
            return self.create(self.val % other)
        else:
            return super().__mod__(other)  # other == 0, 1

    def __pow__(self, other):
        if isinstance(other, ScalarValue):
            return self.create(self.val**other.val)
        elif isinstance(other, self._val_types):
            return self.create(self.val**other)
        else:
            return super().__pow__(other)  # other == 0, 1

    def __radd__(self, other):
        if other == 1:
            return self.create(1 + self.val)
        elif isinstance(other, self._val_types):
            return self.create(other + self.val)
        else:
            return super().__radd__(other)  # other == 0

    def __rsub__(self, other):
        if other == 1:
            return self.create(1 - self.val)
        elif isinstance(other, self._val_types):
            return self.create(other - self.val)
        else:
            return super().__radd__(other)  # other == 0

    def __rmul__(self, other):
        if isinstance(other, self._val_types):
            return self.create(other * self.val)
        else:
            return super().__rmul__(other)  # other == 0, 1

    def __rfloordiv__(self, other):
        if other == 1:
            return self.create(1 // self.val)
        elif isinstance(other, self._val_types):
            return self.create(other // self.val)
        else:
            return super().__rfloordiv__(other)  # other == 0

    def __rtruediv__(self, other):
        if other == 1:
            return self.create(1 / self.val)
        elif isinstance(other, self._val_types):
            return self.create(other / self.val)
        else:
            return super().__rtruediv__(other)  # other == 0, 1/x -> x^(-1)

    def __rmod__(self, other):
        if isinstance(other, self._val_types):
            return self.create(other % self.val)
        else:
            return super().__rmod__(other)  # other == 0

    def __rpow__(self, other):
        if isinstance(other, self._val_types):
            return self.create(other**self.val)
        else:
            return super().__rpow__(other)  # other == 0, 1

    def __complex__(self):
        return complex(self.val)

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def _sympy_(self):
        return sympy.sympify(self.val)


class ScalarExpression(Scalar, metaclass=ABCMeta):
    """Base class for scalars with non-scalar arguments

    For example, a :class:`.BraKet` is a :class:`Scalar`, but has arguments
    that are states.
    """

    _order_index = 1  # Expression scalars come after ScalarValue

    def __pow__(self, other):
        return ScalarPower.create(self, other)


@singleton_object
class Zero(Scalar, metaclass=Singleton):
    """The neutral element with respect to scalar addition

    Equivalent to the scalar value zero::

        >>> Zero == 0
        True
    """

    _order_name = 'ScalarValue'  # sort like ScalarValue(0)
    _hash_val = 0

    @property
    def args(self):
        return tuple()

    @property
    def val(self):
        return self._hash_val

    @property
    def real(self):
        """Real part"""
        return self

    @property
    def imag(self):
        """Imaginary part"""
        return self

    @property
    def _order_key(self):
        return KeyTuple([
            self._order_index, self._order_name or self.__class__.__name__,
            self._order_coeff, KeyTuple([self.val, ]), self._order_kwargs])

    def _diff(self, sym):
        return self

    def _adjoint(self):
        return self

    def __abs__(self):
        return self

    __neg__ = __abs__

    def __lt__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val < other.val
        else:
            return self._hash_val < other

    def __le__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val <= other.val
        else:
            return self._hash_val <= other

    def __gt__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val > other.val
        else:
            return self._hash_val > other

    def __ge__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val >= other.val
        else:
            return self._hash_val >= other

    def __add__(self, other):
        if isinstance(other, ScalarValue):
            return other
        elif isinstance(other, self._val_types):
            return ScalarValue.create(other)
        else:
            return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, ScalarValue):
            return -ScalarValue.create(other)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(-other)
        else:
            return super().__sub__(other)

    def __mul__(self, other):
        try:
            # if possible, keep the type of `other`
            return other._zero
        except AttributeError:
            return self

    def __floordiv__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(0 // other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(0 // other)
        else:
            return super().__floordiv__(other)

    def __truediv__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(0 / other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(0 / other)
        else:
            return super().__truediv__(other)

    def __mod__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(0 % other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(0 % other)
        else:
            return super().__mod__(other)

    def __pow__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(0**other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(0**other)
        else:
            return super().__pow__(other)

    def __radd__(self, other):
        if isinstance(other, self._val_types):
            return ScalarValue.create(other)
        else:
            return super().__radd__(other)

    __rsub__ = __radd__

    def __rmul__(self, other):
        if isinstance(other, self._val_types):
            return self
        else:
            return super().__rmul__(other)

    def __rfloordiv__(self, other):
        if isinstance(other, self._val_types):
            return ScalarValue.create(other // 0)
        else:
            return super().__rfloordiv__(other)

    def __rtruediv__(self, other):
        raise ZeroDivisionError("integer division or modulo by zero")

    __rmod__ = __rtruediv__

    def __rpow__(self, other):
        return One

    def __complex__(self):
        return 0j

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def _sympy_(self):
        return sympy.sympify(0)


@singleton_object
class One(Scalar, metaclass=Singleton):
    """The neutral element with respect to scalar multiplication

    Equivalent to the scalar value one::

        >>> One == 1
        True
    """

    _order_name = 'ScalarValue'  # sort like ScalarValue(1)
    _hash_val = 1

    @property
    def args(self):
        return tuple()

    @property
    def val(self):
        return self._hash_val

    @property
    def real(self):
        """Real part"""
        return self

    @property
    def imag(self):
        """Imaginary part"""
        return Zero

    @property
    def _order_key(self):
        return KeyTuple([
            self._order_index, self._order_name or self.__class__.__name__,
            self._order_coeff, KeyTuple([self.val, ]), self._order_kwargs])

    def _diff(self, sym):
        return Zero

    def __lt__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val < other.val
        else:
            return self._hash_val < other

    def __le__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val <= other.val
        else:
            return self._hash_val <= other

    def __gt__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val > other.val
        else:
            return self._hash_val > other

    def __ge__(self, other):
        if isinstance(other, ScalarValue):
            return self._hash_val >= other.val
        else:
            return self._hash_val >= other

    def _adjoint(self):
        return self

    def __abs__(self):
        return self

    def __neg__(self):
        return ScalarValue(-1)

    def __add__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(1 + other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(1 + other)
        elif other == 1:
            return ScalarValue(2)
        else:
            return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(1 - other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(1 - other)
        elif other == 1:
            return Zero
        else:
            return super().__add__(other)

    def __mul__(self, other):
        if isinstance(other, self._val_types):
            return ScalarValue.create(other)
        else:
            return other

    def __floordiv__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(1 // other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(1 // other)
        else:
            return super().__floordiv__(other)

    def __truediv__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(1 / other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(1 / other)
        else:
            return super().__floordiv__(other)

    def __mod__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(1 % other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(1 % other)
        else:
            return super().__mod__(other)

    def __pow__(self, other):
        if isinstance(other, ScalarValue):
            return ScalarValue.create(1**other.val)
        elif isinstance(other, self._val_types):
            return ScalarValue.create(1**other)
        else:
            return super().__pow__(other)

    def __radd__(self, other):
        if isinstance(other, self._val_types):
            return ScalarValue.create(other + 1)
        else:
            return super().__radd__(other)

    def __rsub__(self, other):
        if isinstance(other, self._val_types):
            return ScalarValue.create(other - 1)
        else:
            return super().__radd__(other)

    __rfloordiv = __rtruediv__ = __rmul__ = __rpow__ = __mul__

    def __rmod__(self, other):
        if isinstance(other, self._val_types):
            return ScalarValue.create(other % 1)
        else:
            return super().__rmod__(other)

    def __complex__(self):
        return 1j

    def __int__(self):
        return 1

    def __float__(self):
        return 1.0

    def _sympy_(self):
        return sympy.sympify(1)


class ScalarPlus(QuantumPlus, Scalar):
    """Sum of scalars

    Generally, :class:`ScalarValue` instances are combined directly::

        >>> alpha = ScalarValue.create(sympy.symbols('alpha'))
        >>> print(srepr(alpha + 1))
        ScalarValue(Add(Symbol('alpha'), Integer(1)))

    An unevaluated :class:`ScalarPlus` remains only for
    :class:`ScalarExpression` instaces::

        >>> braket = KetSymbol('Psi', hs=0).dag() * KetSymbol('Phi', hs=0)
        >>> print(srepr(braket + 1, indented=True))
        ScalarPlus(
            One,
            BraKet(
                KetSymbol(
                    'Psi',
                    hs=LocalSpace(
                        '0')),
                KetSymbol(
                    'Phi',
                    hs=LocalSpace(
                        '0'))))
    """
    _neutral_element = Zero
    _binary_rules = OrderedDict()
    simplifications = [
        assoc, convert_to_scalars, orderby, collect_scalar_summands,
        match_replace_binary]

    def conjugate(self):
        """Complex conjugate of of the sum"""
        return self.__class__.create(
            *[arg.conjugate() for arg in self.args])

    def __pow__(self, other):
        return ScalarPower.create(self, other)


class ScalarTimes(QuantumTimes, Scalar):
    """Product of scalars

    Generally, :class:`ScalarValue` instances are combined directly::

        >>> alpha = ScalarValue.create(sympy.symbols('alpha'))
        >>> print(srepr(alpha * 2))
        ScalarValue(Mul(Integer(2), Symbol('alpha')))

    An unevaluated :class:`ScalarTimes` remains only for
    :class:`ScalarExpression` instaces::

        >>> braket = KetSymbol('Psi', hs=0).dag() * KetSymbol('Phi', hs=0)
        >>> print(srepr(braket * 2, indented=True))
        ScalarTimes(
            ScalarValue(
                2),
            BraKet(
                KetSymbol(
                    'Psi',
                    hs=LocalSpace(
                        '0')),
                KetSymbol(
                    'Phi',
                    hs=LocalSpace(
                        '0'))))
    """
    _neutral_element = One
    _binary_rules = OrderedDict()
    simplifications = [assoc, orderby, filter_neutral, match_replace_binary]

    @classmethod
    def create(cls, *operands, **kwargs):
        """Instantiate the product while applying simplification rules"""
        converted_operands = []
        for op in operands:
            if not isinstance(op, Scalar):
                op = ScalarValue.create(op)
            converted_operands.append(op)
        return super().create(*converted_operands, **kwargs)

    def conjugate(self):
        """Complex conjugate of of the product"""
        return self.__class__.create(
            *[arg.conjugate() for arg in reversed(self.args)])

    def __pow__(self, other):
        return ScalarPower.create(self, other)

    def _expand(self):
        eops = [o.expand() for o in self.operands]

        # store tuples of summands of all expanded factors
        def get_summands(x):
            if isinstance(x, self.__class__._plus_cls):
                return x.operands
            elif isinstance(x, ScalarValue) and isinstance(x.val, sympy.Add):
                return x.val.args
            else:
                return (x, )

        eopssummands = [get_summands(eo) for eo in eops]
        # iterate over a cartesian product of all factor summands, form product
        # of each tuple and sum over result
        summands = []
        for combo in cartesian_product(*eopssummands):
            summand = self.__class__._times_cls.create(*combo)
            summands.append(summand)
        ret = self.__class__._plus_cls.create(*summands)
        if isinstance(ret, self.__class__._plus_cls):
            return ret.expand()
        else:
            return ret


class ScalarIndexedSum(QuantumIndexedSum, Scalar):
    """Indexed sum over scalars"""

    _rules = OrderedDict()
    simplifications = [
        assoc_indexed, indexed_sum_over_kronecker,
        indexed_sum_over_const, match_replace]

    @classmethod
    def create(cls, term, *ranges):
        """Instantiate the indexed sum while applying simplification rules"""
        if not isinstance(term, Scalar):
            term = ScalarValue.create(term)
        return super().create(term, *ranges)

    def __init__(self, term, *ranges):
        if not isinstance(term, Scalar):
            term = ScalarValue.create(term)
        super().__init__(term, *ranges)

    def conjugate(self):
        """Complex conjugate of of the indexed sum"""
        return self.__class__.create(self.term.conjugate(), *self.ranges)

    @property
    def real(self):
        """Real part"""
        return self.__class__.create(self.term.real, *self.ranges)

    @property
    def imag(self):
        """Imaginary part"""
        return self.__class__.create(self.term.imag, *self.ranges)

    def _check_val_type(self, other):
        return (
            isinstance(other, ScalarValue) or
            isinstance(other, Scalar._val_types))

    def __mul__(self, other):
        # For "normal" indexed sums, we prefer to keep scalar factors in front
        # of the sum. For a ScalarIndexedSum, this doesn't make sense, though
        if self._check_val_type(other):
            sum = self
            try:
                idx_syms = [
                    s for s in other.free_symbols if isinstance(s, IdxSym)]
                if len(idx_syms) > 0:
                    sum = self.make_disjunct_indices(*idx_syms)
            except AttributeError:
                pass
            return self.__class__.create(sum.term * other, *self.ranges)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        if self._check_val_type(other):
            sum = self
            try:
                idx_syms = [
                    s for s in other.free_symbols if isinstance(s, IdxSym)]
                if len(idx_syms) > 0:
                    sum = self.make_disjunct_indices(*idx_syms)
            except AttributeError:
                pass
            return self.__class__.create(other * sum.term, *self.ranges)
        else:
            return super().__rmul__(other)

    def __pow__(self, other):
        if other == 0:
            return self._one
        elif other == 1:
            return self
        else:
            try:
                other_is_int = (other == int(other))
            except TypeError:
                other_is_int = False
            if other_is_int:
                if other > 1:
                    res = self
                    for _ in range(other - 1):
                        res = res * self
                    return res
                else:
                    assert other < 1
                    return 1 / self**(-other)
            else:
                return ScalarPower.create(self, other)


class ScalarPower(QuantumOperation, Scalar):
    """A scalar raised to a power

    Generally, :class:`ScalarValue` instances are exponentiated directly::

        >>> alpha = ScalarValue.create(sympy.symbols('alpha'))
        >>> print(srepr(alpha**2))
        ScalarValue(Pow(Symbol('alpha'), Integer(2)))

    An unevaluated :class:`ScalarPower` remains only for
    :class:`ScalarExpression` instaces, see e.g. :func:`sqrt`.
    """

    _rules = OrderedDict()
    simplifications = [convert_to_scalars, match_replace]

    def __init__(self, b, e):
        self._base = b
        self._exp = e
        super().__init__(b, e)

    @property
    def base(self):
        """The base of the exponential"""
        return self._base

    @property
    def exp(self):
        """The exponent"""
        return self._exp

    def __pow__(self, other):
        return ScalarPower.create(self.base, self.exp * other)

    def _adjoint(self):
        return self.__class__.create(
            self._base.conjugate(), self._exp.conjugate())

    def _diff(self, sym):
        inner = self.base._diff(sym)
        if inner != 0:
            return (
                self.exp * ScalarPower.create(self.base, self.exp-1) * inner)
        else:
            return Zero

    def _series_expand(self, param, about, order):
        try:
            if int(self.exp) == self.exp and int(self.exp) > 0:
                # delegate to the _series_expand for ScalarTimes
                prod_ops = [self.base for _ in range(int(self.exp))]
                self_as_product = ScalarTimes(*prod_ops)
                return self_as_product.series_expand(param, about, order)
            else:
                raise ValueError("self.exp is not a positive integer")
        except ValueError:
            # We don't know for sure if self is singular. One way to find out
            # is to substitute symbols for every ScalarExpression (assuming
            # they don't depend on param), let sympy do the series expansion,
            # and then substitute back. We'll keep this option for a later day
            raise ValueError(
                "%s MAY be singular at expansion point %s=%s. Report this "
                "as a bug." % (self, param, about))


class ScalarDerivative(QuantumDerivative, Scalar):
    """Symbolic partial derivative of a scalar

    See :class:`.QuantumDerivative`.
    """
    pass


def KroneckerDelta(i, j, simplify=True):
    """Kronecker delta symbol

    Return :class:`One` (`i` equals `j`)), :class:`Zero` (`i` and `j` are
    non-symbolic an unequal), or a :class:`ScalarValue` wrapping SymPy's
    :class:`~sympy.functions.special.tensor_functions.KroneckerDelta`.

        >>> i, j = IdxSym('i'), IdxSym('j')
        >>> KroneckerDelta(i, i)
        One
        >>> KroneckerDelta(1, 2)
        Zero
        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)

    By default, the Kronecker delta is returned in a simplified form, e.g::

        >>> KroneckerDelta((i+1)/2, (j+1)/2)
        KroneckerDelta(i, j)

    This may be suppressed by setting `simplify` to False::

        >>> KroneckerDelta((i+1)/2, (j+1)/2, simplify=False)
        KroneckerDelta(i/2 + 1/2, j/2 + 1/2)

    Raises:
        TypeError: if `i` or `j` is not an integer or sympy expression. There
        is no automatic sympification of `i` and `j`.
    """
    from qnet.algebra.core.scalar_algebra import ScalarValue, One
    if not isinstance(i, (int, sympy.Basic)):
        raise TypeError(
            "i is not an integer or sympy expression: %s" % type(i))
    if not isinstance(j, (int, sympy.Basic)):
        raise TypeError(
            "j is not an integer or sympy expression: %s" % type(j))
    if i == j:
        return One
    else:
        delta = sympy.KroneckerDelta(i, j)
        if simplify:
            delta = _simplify_delta(delta)
        return ScalarValue.create(delta)


def sqrt(scalar):
    """Square root of a :class:`Scalar` or scalar value

    This always returns a :class:`Scalar`, and uses a symbolic square root if
    possible (i.e., for non-floats)::

        >>> sqrt(2)
        sqrt(2)

        >>> sqrt(2.0)
        1.414213...

    For a :class:`ScalarExpression` argument, it returns a
    :class:`ScalarPower` instance::

        >>> braket = KetSymbol('Psi', hs=0).dag() * KetSymbol('Phi', hs=0)
        >>> nrm = sqrt(braket * braket.dag())
        >>> print(srepr(nrm, indented=True))
        ScalarPower(
            ScalarTimes(
                BraKet(
                    KetSymbol(
                        'Phi',
                        hs=LocalSpace(
                            '0')),
                    KetSymbol(
                        'Psi',
                        hs=LocalSpace(
                            '0'))),
                BraKet(
                    KetSymbol(
                        'Psi',
                        hs=LocalSpace(
                            '0')),
                    KetSymbol(
                        'Phi',
                        hs=LocalSpace(
                            '0')))),
            ScalarValue(
                Rational(1, 2)))
    """
    if isinstance(scalar, ScalarValue):
        scalar = scalar.val
    if scalar == 1:
        return One
    elif scalar == 0:
        return Zero
    elif isinstance(scalar, (float, complex, complex128, float64)):
        return ScalarValue.create(numpy.sqrt(scalar))
    elif isinstance(scalar, (int, sympy.Basic, int64)):
        return ScalarValue.create(sympy.sqrt(scalar))
    elif isinstance(scalar, Scalar):
        return scalar**(sympy.sympify(1) / 2)
    else:
        raise TypeError("Unknown type of scalar: %r" % type(scalar))


def is_scalar(scalar):
    """Check if `scalar` is a :class:`Scalar` or a scalar value

    Specifically, whether `scalar` is an instance of :class:`Scalar` or an
    instance of a numeric or symbolic type that could be wrapped in
    :class:`ScalarValue`.

    For internal use only.
    """
    return isinstance(scalar, Scalar) or isinstance(scalar, Scalar._val_types)


Scalar._zero = Zero
Scalar._one = One
Scalar._base_cls = Scalar
Scalar._scalar_times_expr_cls = ScalarTimes
Scalar._plus_cls = ScalarPlus
Scalar._times_cls = ScalarTimes
Scalar._adjoint_cls = lambda scalar: scalar.conjugate()
Scalar._adjoint_cls.create = Scalar._adjoint_cls  # mock Expression
Scalar._indexed_sum_cls = ScalarIndexedSum
Scalar._derivative_cls = ScalarDerivative
