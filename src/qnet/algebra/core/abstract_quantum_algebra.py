"""Common algebra of "quantum" objects

Quantum objects have an associated Hilbert space, and they (at least partially)
summation, products, multiplication with a scalar, and adjoints.

The algebra defined in this module is the superset of the Hilbert space algebra
of states (augmented by the tensor product), and the C* algebras of operators
and superoperators.
"""
import re
from abc import ABCMeta, abstractmethod
from itertools import product as cartesian_product

from sympy import (
    Basic as SympyBasic, Expr as SympyExpr, Symbol, sympify, series as
    sympy_series, diff as sympy_diff)

from .scalar_types import SCALAR_TYPES
from .hilbert_space_algebra import ProductSpace, LocalSpace, TrivialSpace
from .abstract_algebra import (
    Operation, Expression,
    substitute, _scalar_free_symbols, )
from .indexed_operations import IndexedSum
from ...utils.ordering import (
    DisjunctCommutativeHSOrder, FullCommutativeHSOrder, KeyTuple, )
from ...utils.indices import SymbolicLabelBase


__all__ = [
    'ScalarTimesQuantumExpression', 'QuantumExpression', 'QuantumOperation',
    'QuantumPlus', 'QuantumTimes', 'SingleQuantumOperation', 'QuantumAdjoint',
    'QuantumSymbol', 'QuantumIndexedSum']
__private__ = [
    'ensure_local_space']


_sympyOne = sympify(1)


def _simplify_scalar(s):
    """Simplify all occurences of scalar expressions in s

    Args:
        s (Expression or sympy.core.basic.Basic): The expression to simplify.

    Returns:
        A simplified expression of the same type as `s`.
    """
    # TODO: this routine would be obsolete if we had a proper Scalar class
    try:
        return s.simplify_scalar()
    except AttributeError:
        pass
    if isinstance(s, SympyExpr):
        return s.simplify()
    return s


class QuantumExpression(Expression, metaclass=ABCMeta):
    """Common base class for any expression that is associated with a Hilbert
    space"""

    _zero = None  # The neutral element for addition
    _one = None  # The neutral element for multiplication
    _base_cls = None  # The most general class we can add / multiply
    _scalar_times_expr_cls = None   # class for multiplication with scalar
    _plus_cls = None  # class for internal addition
    _times_cls = None  # class for internal multiplication
    _adjoint_cls = None  # class for the adjoint
    _indexed_sum_cls = None  # class for indexed sum

    _order_index = 0
    _order_coeff = 1
    _order_name = None

    def __init__(self, *args, **kwargs):
        self._order_args = KeyTuple([
            arg._order_key if hasattr(arg, '_order_key') else arg
            for arg in args])
        self._order_kwargs = KeyTuple([
            KeyTuple([
                key, val._order_key if hasattr(val, '_order_key') else val])
            for (key, val) in sorted(kwargs.items())])
        super().__init__(*args, **kwargs)

    @property
    def _order_key(self):
        return KeyTuple([
            self._order_index, self._order_name or self.__class__.__name__,
            self._order_coeff, self._order_args, self._order_kwargs])

    @property
    @abstractmethod
    def space(self):
        """The :class:`.HilbertSpace` on which the operator acts
        non-trivially"""
        raise NotImplementedError(self.__class__.__name__)

    def adjoint(self):
        """The Hermitian adjoint of the Expression"""
        return self._adjoint()

    def dag(self):
        """Alias for :meth:`adjoint`"""
        return self._adjoint()

    def conjugate(self):
        """Alias for :meth:`adjoint`"""
        return self._adjoint()

    @abstractmethod
    def _adjoint(self):
        raise NotImplementedError(self.__class__.__name__)

    def expand(self):
        """Expand out distributively all products of sums. Note that this does
        not expand out sums of scalar coefficients.
        """
        return self._expand()

    def _expand(self):
        return self

    def simplify_scalar(self):
        """Simplify all scalar coefficients within the expression.

        Returns:
            Operator: The simplified expression.
        """
        return self._simplify_scalar()

    def _simplify_scalar(self):
        return self

    def diff(self, sym: Symbol, n: int = 1, expand_simplify: bool = True):
        """Differentiate by scalar parameter `sym`.

        Args:
            sym: What to differentiate by.
            n: How often to differentiate
            expand_simplify: Whether to simplify the result.

        Returns:
            The n-th derivative.
        """
        expr = self
        for k in range(n):
            expr = expr._diff(sym)
        if expand_simplify:
            expr = expr.expand().simplify_scalar()
        return expr

    def _diff(self, sym):
        # Expressions are assumed constant by default.
        return self.__class__._zero

    def series_expand(
            self, param: Symbol, about, order: int) -> tuple:
        """Expand the operator expression as a truncated power series in a
        scalar parameter.

        Args:
            param: Expansion parameter
            about (SCALAR_TYPES): Point about which to expand
            order: Maximum order of expansion (>= 0)

        Returns:
            tuple of length ``order + 1``, where the entries are the
            expansion coefficients (instances of :class:`Operator`)
        """
        return self._series_expand(param, about, order)

    def _series_expand(self, param, about, order):
        # Expressions are assumed constant by default.
        return (self,) + ((0,) * order)

    def __add__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self.__class__._plus_cls.create(
                self, other * self.__class__._one)
        elif isinstance(other, self.__class__._base_cls):
            return self.__class__._plus_cls.create(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        # addition is assumed to be commutative
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self.__class__._scalar_times_expr_cls.create(other, self)
        elif isinstance(other, self.__class__._base_cls):
            return self.__class__._times_cls.create(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        # multiplication with scalar is assumed to be commutative, but any
        # other multiplication is not
        if isinstance(other, SCALAR_TYPES):
            return self.__class__._scalar_times_expr_cls.create(other, self)
        else:
            return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self * (_sympyOne / other)
        try:
            return super().__rmul__(other)
        except AttributeError:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, int):
            return self.__class__._times_cls.create(
                *[self for _ in range(other)])
        else:
            return NotImplemented


class QuantumSymbol(QuantumExpression, metaclass=ABCMeta):
    """A symbolic expression"""
    _rx_label = re.compile('^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9().+-]+)?$')

    def __init__(self, label, *, hs):
        self._label = label
        if isinstance(label, str):
            if not self._rx_label.match(label):
                raise ValueError(
                    "label '%s' does not match pattern '%s'"
                    % (label, self._rx_label.pattern))
        elif isinstance(label, SymbolicLabelBase):
            self._label = label
        else:
            raise TypeError(
                "type of label must be str or SymbolicLabelBase, not %s"
                % type(label))
        if isinstance(hs, (str, int)):
            hs = LocalSpace(hs)
        elif isinstance(hs, tuple):
            hs = ProductSpace.create(*[LocalSpace(h) for h in hs])
        self._hs = hs
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
        try:
            return self.label.all_symbols()
        except AttributeError:
            return set([])

    def _adjoint(self):
        return self.__class__._adjoint_cls(self)


class QuantumOperation(QuantumExpression, Operation, metaclass=ABCMeta):
    """Base class for operations on quantum expressions within the same
    fundamental set"""

    # "same fundamental set" means all operandas are instances of _base_cls
    # Operations that involve objects from different sets should directly
    # subclass from QuantumExpression and Operation

    _order_index = 1  # Operations are printed after "atomic" Expressions

    def __init__(self, *operands, **kwargs):
        for o in operands:
            assert isinstance(o, self.__class__._base_cls)
        op_spaces = [o.space for o in operands]
        self._space = ProductSpace.create(*op_spaces)
        super().__init__(*operands, **kwargs)

    @property
    def space(self):
        """Hilbert space of the operation result"""
        return self._space

    def _simplify_scalar(self):
        return self.create(*[o.simplify_scalar() for o in self.operands])


class SingleQuantumOperation(QuantumOperation, metaclass=ABCMeta):
    """Base class for operations on a single quantum expression"""

    def __init__(self, op, **kwargs):
        if isinstance(op, SCALAR_TYPES):
            op = op * self.__class__._one
        super().__init__(op, **kwargs)

    @property
    def operand(self):
        """The operator that the operation acts on"""
        return self.operands[0]

    def _series_expand(self, param, about, order):
        ope = self.operand.series_expand(param, about, order)
        return tuple(opet.adjoint() for opet in ope)


class QuantumAdjoint(SingleQuantumOperation, metaclass=ABCMeta):
    """Base class for adjoints of quantum expressions"""

    def _expand(self):
        eo = self.operand.expand()
        if isinstance(eo, self.__class__._plus_cls):
            summands = [eoo.adjoin() for eoo in eo.operands]
            return self.__class__._plus_cls.create(*summands)
        return eo.adjoint()

    def _diff(self, sym):
        return self.__class__.create(self.operands[0]._diff(sym))

    def _adjoint(self):
        return self.operand


class QuantumPlus(QuantumOperation, metaclass=ABCMeta):
    """General implementation of addition of quantum expressions"""
    order_key = FullCommutativeHSOrder
    neutral_element = None

    def _expand(self):
        summands = [o.expand() for o in self.operands]
        return self.__class__._plus_cls.create(*summands)

    def _series_expand(self, param, about, order):
        tuples = (o.series_expand(param, about, order) for o in self.operands)
        res = (self.__class__._plus_cls.create(*tels) for tels in zip(*tuples))
        return res

    def _diff(self, sym):
        return sum([o._diff(sym) for o in self.operands], self.__class__._zero)

    def _adjoint(self):
        return self.__class__._plus_cls(*[o.adjoint() for o in self.operands])


class QuantumTimes(QuantumOperation, metaclass=ABCMeta):
    """General implementation of product of quantum expressions"""
    order_key = DisjunctCommutativeHSOrder
    neutral_element = None

    def factor_for_space(self, spc):
        """Return a tuple of two products, where the first product contains the
        given Hilbert space, and the second product is disjunct from it."""
        if spc == TrivialSpace:
            ops_on_spc = [
                o for o in self.operands if o.space is TrivialSpace]
            ops_not_on_spc = [
                o for o in self.operands if o.space > TrivialSpace]
        else:
            ops_on_spc = [
                o for o in self.operands if (o.space & spc) > TrivialSpace]
            ops_not_on_spc = [
                o for o in self.operands if (o.space & spc) is TrivialSpace]
        return (
            self.__class__._times_cls.create(*ops_on_spc),
            self.__class__._times_cls.create(*ops_not_on_spc))

    def _expand(self):
        eops = [o.expand() for o in self.operands]
        # store tuples of summands of all expanded factors
        eopssummands = [
            eo.operands if isinstance(eo, self.__class__._plus_cls) else (eo,)
            for eo in eops]
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

    def _series_expand(self, param, about, order):
        assert len(self.operands) > 1
        cfirst = self.operands[0].series_expand(param, about, order)
        crest = (
            self.__class__._times_cls.create(*self.operands[1:])
            .series_expand(param, about, order))
        res = []
        for n in range(order + 1):
            summands = [cfirst[k] * crest[n - k] for k in range(n + 1)]
            res.append(self.__class__._plus_cls.create(*summands))
        return tuple(res)

    def _diff(self, sym):
        assert len(self.operands) > 1
        first = self.operands[0]
        rest = self.__class__._times_cls.create(*self.operands[1:])
        return first._diff(sym) * rest + first * rest._diff(sym)

    def _adjoint(self):
        return self.__class__._times_cls.create(
                *[o.adjoint() for o in reversed(self.operands)])


class ScalarTimesQuantumExpression(
        QuantumExpression, Operation, metaclass=ABCMeta):
    """Product of a scalar and an expression"""

    def __init__(self, coeff, term):
        self._order_coeff = coeff
        self._order_args = KeyTuple([term._order_key])
        super().__init__(coeff, term)

    @property
    def coeff(self):
        return self.operands[0]

    @property
    def term(self):
        return self.operands[1]

    def _substitute(self, var_map, safe=False):
        st = self.term.substitute(var_map)
        if isinstance(self.coeff, SympyBasic):
            svar_map = {k: v for k, v in var_map.items()
                        if not isinstance(k, Expression)}
            sc = self.coeff.subs(svar_map)
        else:
            sc = substitute(self.coeff, var_map)
        if safe:
            return self.__class__(sc, st)
        else:
            return sc * st

    def all_symbols(self):
        return _scalar_free_symbols(self.coeff) | self.term.all_symbols()

    def _adjoint(self):
        return self.coeff.conjugate() * self.term.adjoint()

    @property
    def _order_key(self):
        from qnet.printing.asciiprinter import QnetAsciiDefaultPrinter
        ascii = QnetAsciiDefaultPrinter().doprint
        t = self.term._order_key
        try:
            c = abs(float(self.coeff))  # smallest coefficients first
        except (ValueError, TypeError):
            c = float('inf')
        return KeyTuple(t[:2] + (c,) + t[3:] + (ascii(self.coeff),))

    @property
    def space(self):
        return self.operands[1].space

    def _expand(self):
        c, t = self.operands
        et = t.expand()
        if isinstance(et, self.__class__._plus_cls):
            summands = [c * eto for eto in et.operands]
            return self.__class__._plus_cls.create(*summands)
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
                    raise ValueError(
                        "%s is singular at expansion point %s=%s."
                        % (self, param, about))
                if o > order:
                    break
                ce.extend([0] * (o - next_order))
                ce.append(c)
                next_order = o + 1
            ce.extend([0] * (order + 1 - next_order))

            res = []
            for n in range(order + 1):
                summands = [ce[k] * te[n - k] for k in range(n + 1)]
                res.append(self.__class__._plus_cls.create(*summands))
            return tuple(res)
        else:
            return tuple(self.coeff * tek for tek in te)

    def _diff(self, sym):
        c, t = self.operands
        cd = sympy_diff(c, sym) if isinstance(c, SympyBasic) else 0
        return cd * t + c * t._diff(sym)

    def _simplify_scalar(self):
        coeff, term = self.operands
        return _simplify_scalar(coeff) * term.simplify_scalar()

    def __complex__(self):
        if self.term is self.__class__._one:
            return complex(self.coeff)
        return NotImplemented

    def __float__(self):
        if self.term is self.__class__._one:
            return float(self.coeff)
        return NotImplemented


class QuantumIndexedSum(IndexedSum, SingleQuantumOperation, metaclass=ABCMeta):

    @property
    def space(self):
        """The Hilbert space of the sum's term"""
        return self.term.space

    def _expand(self):
        return self.__class__.create(self.term.expand(), *self.ranges)

    def _series_expand(self, param, about, order):
        raise NotImplementedError()

    def _adjoint(self):
        return self.__class__.create(self.term.adjoint(), *self.ranges)

    def __mul__(self, other):
        if isinstance(other, IndexedSum):
            other = other.make_disjunct_indices(self)
            new_ranges = self.ranges + other.ranges
            new_term = self.term * other.term
            # note that class may change, depending on type of new_term
            return new_term.__class__._indexed_sum_cls.create(
                new_term, *new_ranges)
        elif isinstance(other, SCALAR_TYPES):
            return self._class__._scalar_times_expr_cls(other, self)
        elif isinstance(other, ScalarTimesQuantumExpression):
            return self._class__._scalar_times_expr_cls(
                other.coeff, self * other.term)
        else:
            # TODO: ensure other has no bound symbols that overlap with
            # self.term
            new_term = self.term * other
            return new_term.__class__._indexed_sum_cls.create(
                new_term, *self.ranges)

    def __rmul__(self, other):
        if isinstance(other, IndexedSum):
            self_new = self.make_disjunct_indices(other)
            new_ranges = other.ranges + self_new.ranges
            new_term = other.term * self_new.term
            # note that class may change, depending on type of new_term
            return new_term.__class__._indexed_sum_cls.create(
                new_term, *new_ranges)
        elif isinstance(other, SCALAR_TYPES):
            return self.__class__._scalar_times_expr_cls(other, self)
        elif isinstance(other, ScalarTimesQuantumExpression):
            return self._class__._scalar_times_expr_cls(
                other.coeff, other.term * self)
        else:
            new_term = other * self.term
            return new_term.__class__._indexed_sum_cls.create(
                new_term, *self.ranges)

    def __add__(self, other):
        raise NotImplementedError()

    def __radd__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        raise NotImplementedError()

    def __rsub__(self, other):
        raise NotImplementedError()


def ensure_local_space(hs):
    """Ensure that the given `hs` is an instance of :class:`.LocalSpace`.

    If `hs` an instance of :class:`str` or :class:`int`, it will be converted
    to a :class:`.LocalSpace`. If it already is a :class:`.LocalSpace`, `hs`
    will be returned unchanged.

    Raises:
        TypeError: If `hs` is not a :class:`.LocalSpace`, :class:`str`, or
            :class:`int`.

    Returns:
        LocalSpace: original or converted `hs`

    Examples:
        >>> srepr(ensure_local_space(0))
        "LocalSpace('0')"
        >>> srepr(ensure_local_space('tls'))
        "LocalSpace('tls')"
        >>> srepr(ensure_local_space(LocalSpace(0)))
        "LocalSpace('0')"
        >>> srepr(ensure_local_space(LocalSpace(0) * LocalSpace(1)))
        Traceback (most recent call last):
           ...
        TypeError: hs must be a LocalSpace
    """
    if isinstance(hs, (str, int)):
        hs = LocalSpace(hs)
    if not isinstance(hs, LocalSpace):
        raise TypeError("hs must be a LocalSpace")
    return hs
