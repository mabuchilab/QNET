import logging
import itertools
from functools import partial

from collections import OrderedDict
import sympy
from sympy.concrete.delta import (
    _has_simple_delta, _extract_delta as _sympy_extract_delta)

from .abstract_algebra import LOG, LEVEL, LOG_NO_MATCH
from .exceptions import CannotSimplify
from ..pattern_matching import ProtoExpr, Pattern, match_pattern
from ...utils.indices import IdxSym

__all__ = []
__private__ = [
    'assoc', 'assoc_indexed', 'idem', 'orderby', 'filter_neutral',
    'filter_cid', 'match_replace', 'match_replace_binary', 'check_cdims',
    'convert_to_spaces', 'empty_trivial', 'implied_local_space',
    'delegate_to_method', 'scalars_to_op', 'convert_to_scalars',
    'disjunct_hs_zero', 'commutator_order', 'accept_bras',
    'basis_ket_zero_outside_hs', 'indexed_sum_over_const',
    'indexed_sum_over_kronecker', 'derivative_via_diff', 'collect_summands',
    'collect_scalar_summands']


_RESOLVE_KRONECKER_WITH_PIECEWISE = False
# Handling indexed sums over Kronecker deltas correctly in the most general
# cases requires substituting it with a Piecewise function (the delta is zero
# outside of the range covered by the summmation index). This incurrs
# considerable numerical overhead. In many cases, one gets correct results
# while ignoring the range of the summation, with less effort. In that case,
# you may speed calculations by settings this flag to False, at your own risk.
#
# An exmple where one gets the wrong result by ignoring the summation range is
# this:
#
#   i, j = symbols('i, j', cls=IdxSym)
#   sum = Sum(i, (1, 2, 3))(Sum(j, (3, 4))(KroneckerDelta(i, j)))
#
#   The wrong result is 3, the correct result is 1


def assoc(cls, ops, kwargs):
    """Associatively expand out nested arguments of the flat class.
    E.g.::

        >>> class Plus(Operation):
        ...     simplifications = [assoc, ]
        >>> Plus.create(1,Plus(2,3))
        Plus(1, 2, 3)
    """
    expanded = [(o,) if not isinstance(o, cls) else o.operands for o in ops]
    return sum(expanded, ()), kwargs


def assoc_indexed(cls, ops, kwargs):
    r"""Flatten nested indexed structures while pulling out possible prefactors

    For example, for an :class:`.IndexedSum`:

    .. math::

        \sum_j \left( a \sum_i \dots \right) = a \sum_{j, i} \dots
    """
    from qnet.algebra.core.abstract_quantum_algebra import (
        ScalarTimesQuantumExpression)
    term, *ranges = ops

    if isinstance(term, cls):
        coeff = 1
    elif isinstance(term, ScalarTimesQuantumExpression):
        coeff = term.coeff
        term = term.term
        if not isinstance(term, cls):
            return ops, kwargs
    else:
        return ops, kwargs

    term = term.make_disjunct_indices(*ranges)
    combined_ranges = tuple(ranges) + term.ranges

    if coeff == 1:
        return cls.create(term.term, *combined_ranges)
    else:
        bound_symbols = set([r.index_symbol for r in combined_ranges])
        if len(coeff.free_symbols.intersection(bound_symbols)) == 0:
            return coeff * cls.create(term.term, *combined_ranges)
        else:
            return cls.create(coeff * term.term, *combined_ranges)


def idem(cls, ops, kwargs):
    """Remove duplicate arguments and order them via the cls's order_key key
    object/function.
    E.g.::

        >>> class Set(Operation):
        ...     order_key = lambda val: val
        ...     simplifications = [idem, ]
        >>> Set.create(1,2,3,1,3)
        Set(1, 2, 3)
    """
    return sorted(set(ops), key=cls.order_key), kwargs


def orderby(cls, ops, kwargs):
    """Re-order arguments via the class's ``order_key`` key object/function.
    Use this for commutative operations:
    E.g.::

        >>> class Times(Operation):
        ...     order_key = lambda val: val
        ...     simplifications = [orderby, ]
        >>> Times.create(2,1)
        Times(1, 2)
    """
    return sorted(ops, key=cls.order_key), kwargs


def filter_neutral(cls, ops, kwargs):
    """Remove occurrences of a neutral element from the argument/operand list,
    if that list has at least two elements.  To use this, one must also specify
    a neutral element, which can be anything that allows for an equality check
    with each argument.  E.g.::

        >>> class X(Operation):
        ...     _neutral_element = 1
        ...     simplifications = [filter_neutral, ]
        >>> X.create(2,1,3,1)
        X(2, 3)
    """
    c_n = cls._neutral_element
    if len(ops) == 0:
        return c_n
    fops = [op for op in ops if c_n != op]  # op != c_n does NOT work
    if len(fops) > 1:
        return fops, kwargs
    elif len(fops) == 1:
        # the remaining operand is the single non-trivial one
        return fops[0]
    else:
        # the original list of operands consists only of neutral elements
        return ops[0]


def collect_summands(cls, ops, kwargs):
    """Collect summands that occur multiple times into a single summand

    Also filters out zero-summands.

    Example:
        >>> A, B, C = (OperatorSymbol(s, hs=0) for s in ('A', 'B', 'C'))
        >>> collect_summands(
        ...     OperatorPlus, (A, B, C, ZeroOperator, 2 * A, B, -C) , {})
        ((3 * A^(0), 2 * B^(0)), {})
        >>> collect_summands(OperatorPlus, (A, -A), {})
        ZeroOperator
        >>> collect_summands(OperatorPlus, (B, A, -B), {})
        A^(0)
    """
    from qnet.algebra.core.abstract_quantum_algebra import (
        ScalarTimesQuantumExpression)
    coeff_map = OrderedDict()
    for op in ops:
        if isinstance(op, ScalarTimesQuantumExpression):
            coeff, term = op.coeff, op.term
        else:
            coeff, term = 1, op
        if term in coeff_map:
            coeff_map[term] += coeff
        else:
            coeff_map[term] = coeff
    fops = []
    for (term, coeff) in coeff_map.items():
        op = coeff * term
        if not op.is_zero:
            fops.append(op)
    if len(fops) == 0:
        return cls._zero
    elif len(fops) == 1:
        return fops[0]
    else:
        return tuple(fops), kwargs


def collect_scalar_summands(cls, ops, kwargs):
    """Collect :class:`ValueScalar` and :class:`ScalarExpression` summands

    Example:
        >>> srepr(collect_scalar_summands(Scalar, (1, 2, 3), {}))
        'ScalarValue(6)'
        >>> collect_scalar_summands(Scalar, (1, 1, -1), {})
        One
        >>> collect_scalar_summands(Scalar, (1, -1), {})
        Zero

        >>> Psi = KetSymbol("Psi", hs=0)
        >>> Phi = KetSymbol("Phi", hs=0)
        >>> braket = BraKet.create(Psi, Phi)

        >>> collect_scalar_summands(Scalar, (1, braket, -1), {})
        <Psi|Phi>^(0)
        >>> collect_scalar_summands(Scalar, (1, 2 * braket, 2, 2 * braket), {})
        ((3, 4 * <Psi|Phi>^(0)), {})
        >>> collect_scalar_summands(Scalar, (2 * braket, -braket, -braket), {})
        Zero
    """
    # This routine is required because there is no
    # "ScalarTimesQuantumExpression" for scalars: we have to extract
    # coefficiencts from ScalarTimes instead
    from qnet.algebra.core.scalar_algebra import (
        Zero, One, Scalar, ScalarTimes, ScalarValue)
    a_0 = Zero
    coeff_map = OrderedDict()
    for op in ops:
        if isinstance(op, ScalarValue) or isinstance(op, Scalar._val_types):
            a_0 += op
            continue
        elif isinstance(op, ScalarTimes):
            if isinstance(op.operands[0], ScalarValue):
                coeff = op.operands[0]
                term = op.operands[1]
                for sub_op in op.operands[2:]:
                    term *= sub_op
            else:
                coeff, term = One, op
        else:
            coeff, term = One, op
        if term in coeff_map:
            coeff_map[term] += coeff
        else:
            coeff_map[term] = coeff
    if a_0 == Zero:
        fops = []
    else:
        fops = [a_0]
    for (term, coeff) in coeff_map.items():
        op = coeff * term
        if not op.is_zero:
            fops.append(op)
    if len(fops) == 0:
        return cls._zero
    elif len(fops) == 1:
        return fops[0]
    else:
        return tuple(fops), kwargs


def match_replace(cls, ops, kwargs):
    """Match and replace a full operand specification to a function that
    provides a replacement for the whole expression
    or raises a :exc:`.CannotSimplify` exception.
    E.g.

    First define an operation::

        >>> class Invert(Operation):
        ...     _rules = OrderedDict()
        ...     simplifications = [match_replace, ]

    Then some _rules::

        >>> A = wc("A")
        >>> A_float = wc("A", head=float)
        >>> Invert_A = pattern(Invert, A)
        >>> Invert._rules.update([
        ...     ('r1', (pattern_head(Invert_A), lambda A: A)),
        ...     ('r2', (pattern_head(A_float), lambda A: 1./A)),
        ... ])

    Check rule application::

        >>> print(srepr(Invert.create("hallo")))  # matches no rule
        Invert('hallo')
        >>> Invert.create(Invert("hallo"))        # matches first rule
        'hallo'
        >>> Invert.create(.2)                     # matches second rule
        5.0

    A pattern can also have the same wildcard appear twice::

        >>> class X(Operation):
        ...     _rules = {
        ...         'r1': (pattern_head(A, A), lambda A: A),
        ...     }
        ...     simplifications = [match_replace, ]
        >>> X.create(1,2)
        X(1, 2)
        >>> X.create(1,1)
        1

    """
    expr = ProtoExpr(ops, kwargs)
    if LOG:
        logger = logging.getLogger('QNET.create')
    for key, rule in cls._rules.items():
        pat, replacement = rule
        match_dict = match_pattern(pat, expr)
        if match_dict:
            try:
                replaced = replacement(**match_dict)
                if LOG:
                    logger.debug(
                        "%sRule %s.%s: (%s, %s) -> %s", ("  " * (LEVEL)),
                        cls.__name__, key, expr.args, expr.kwargs, replaced)
                return replaced
            except CannotSimplify:
                if LOG_NO_MATCH:
                    logger.debug(
                        "%sRule %s.%s: no match: CannotSimplify",
                        ("  " * (LEVEL)), cls.__name__, key)
                continue
        else:
            if LOG_NO_MATCH:
                logger.debug(
                    "%sRule %s.%s: no match: %s", ("  " * (LEVEL)),
                    cls.__name__, key, match_dict.reason)
    # No matching rules
    return ops, kwargs


def _get_binary_replacement(first, second, cls):
    """Helper function for match_replace_binary"""
    expr = ProtoExpr([first, second], {})
    if LOG:
        logger = logging.getLogger('QNET.create')
    for key, rule in cls._binary_rules.items():
        pat, replacement = rule
        match_dict = match_pattern(pat, expr)
        if match_dict:
            try:
                replaced = replacement(**match_dict)
                if LOG:
                    logger.debug(
                        "%sRule %s.%s: (%s, %s) -> %s", ("  " * (LEVEL)),
                        cls.__name__, key, expr.args, expr.kwargs, replaced)
                return replaced
            except CannotSimplify:
                if LOG_NO_MATCH:
                    logger.debug(
                        "%sRule %s.%s: no match: CannotSimplify",
                        ("  " * (LEVEL)), cls.__name__, key)
                continue
        else:
            if LOG_NO_MATCH:
                logger.debug(
                    "%sRule %s.%s: no match: %s", ("  " * (LEVEL)),
                    cls.__name__, key, match_dict.reason)
    return None


def match_replace_binary(cls, ops, kwargs):
    """Similar to func:`match_replace`, but for arbitrary length operations,
    such that each two pairs of subsequent operands are matched pairwise.

        >>> A = wc("A")
        >>> class FilterDupes(Operation):
        ...     _binary_rules = {
        ...          'filter_dupes': (pattern_head(A,A), lambda A: A)}
        ...     simplifications = [match_replace_binary, assoc]
        ...     _neutral_element = 0
        >>> FilterDupes.create(1,2,3,4)         # No duplicates
        FilterDupes(1, 2, 3, 4)
        >>> FilterDupes.create(1,2,2,3,4)       # Some duplicates
        FilterDupes(1, 2, 3, 4)

    Note that this only works for *subsequent* duplicate entries:

        >>> FilterDupes.create(1,2,3,2,4)       # No *subsequent* duplicates
        FilterDupes(1, 2, 3, 2, 4)

    Any operation that uses binary reduction must be associative and define a
    neutral element. The binary rules must be compatible with associativity,
    i.e. there is no specific order in which the rules are applied to pairs of
    operands.
    """
    assert assoc in cls.simplifications, (
        cls.__name__ + " must be associative to use match_replace_binary")
    assert hasattr(cls, '_neutral_element'), (
        cls.__name__ + " must define a neutral element to use "
                       "match_replace_binary")
    fops = _match_replace_binary(cls, list(ops))
    if len(fops) == 1:
        return fops[0]
    elif len(fops) == 0:
        return cls._neutral_element
    else:
        return fops, kwargs


def _match_replace_binary(cls, ops: list) -> list:
    """Reduce list of `ops`"""
    n = len(ops)
    if n <= 1:
        return ops
    ops_left = ops[:n // 2]
    ops_right = ops[n // 2:]
    return _match_replace_binary_combine(
        cls,
        _match_replace_binary(cls, ops_left),
        _match_replace_binary(cls, ops_right))


def _match_replace_binary_combine(cls, a: list, b: list) -> list:
    """combine two fully reduced lists a, b"""
    if len(a) == 0 or len(b) == 0:
        return a + b
    r = _get_binary_replacement(a[-1], b[0], cls)
    if r is None:
        return a + b
    if r == cls._neutral_element:
        return _match_replace_binary_combine(cls, a[:-1], b[1:])
    if isinstance(r, cls):
        r = list(r.args)
    else:
        r = [r, ]
    return _match_replace_binary_combine(
        cls,
        _match_replace_binary_combine(cls, a[:-1], r),
        b[1:])


def check_cdims(cls, ops, kwargs):
    """Check that all operands (`ops`) have equal channel dimension."""
    if not len({o.cdim for o in ops}) == 1:
        raise ValueError("Not all operands have the same cdim:" + str(ops))
    return ops, kwargs


def filter_cid(cls, ops, kwargs):
    """Remove occurrences of the :func:`.circuit_identity` ``cid(n)`` for any
    ``n``. Cf. :func:`filter_neutral`
    """
    from qnet.algebra.core.circuit_algebra import CircuitZero, circuit_identity
    if len(ops) == 0:
        return CircuitZero
    fops = [op for op in ops if op != circuit_identity(op.cdim)]
    if len(fops) > 1:
        return fops, kwargs
    elif len(fops) == 1:
        # the remaining operand is the single non-trivial one
        return fops[0]
    else:
        # the original list of operands consists only of neutral elements
        return ops[0]


def convert_to_spaces(cls, ops, kwargs):
    """For all operands that are merely of type str or int, substitute
    LocalSpace objects with corresponding labels:
    For a string, just itself, for an int, a string version of that int.
    """
    from qnet.algebra.core.hilbert_space_algebra import (
        HilbertSpace, LocalSpace)
    cops = [o if isinstance(o, HilbertSpace) else LocalSpace(o) for o in ops]
    return cops, kwargs


def empty_trivial(cls, ops, kwargs):
    """A ProductSpace of zero Hilbert spaces should yield the TrivialSpace"""
    from qnet.algebra.core.hilbert_space_algebra import TrivialSpace
    if len(ops) == 0:
        return TrivialSpace
    else:
        return ops, kwargs


def implied_local_space(*, arg_index=None, keys=None):
    """Return a simplification that converts the positional argument
    `arg_index` from (str, int) to a subclass of :class:`.LocalSpace`, as well
    as any keyword argument with one of the given keys.

    The exact type of the resulting Hilbert space is determined by
    the `default_hs_cls` argument of :func:`init_algebra`.

    In many cases, we have :func:`implied_local_space` (in ``create``) in
    addition to a conversion in ``__init__``, so
    that :func:`match_replace` etc can rely on the relevant arguments being a
    :class:`HilbertSpace` instance.
    """
    from qnet.algebra.core.hilbert_space_algebra import (
        HilbertSpace, LocalSpace)

    def args_to_local_space(cls, args, kwargs):
        """Convert (str, int) of selected args to :class:`.LocalSpace`"""
        if isinstance(args[arg_index], LocalSpace):
            new_args = args
        else:
            if isinstance(args[arg_index], (int, str)):
                try:
                    hs = cls._default_hs_cls(args[arg_index])
                except AttributeError:
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
                        try:
                            val = cls._default_hs_cls(val)
                        except AttributeError:
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
    from qnet.algebra.core.scalar_algebra import is_scalar
    op_ops = []
    for op in ops:
        if is_scalar(op):
            op_ops.append(op * cls._one)
        else:
            op_ops.append(op)
    return op_ops, kwargs


def convert_to_scalars(cls, ops, kwargs):
    """Convert any entry in `ops` that is not a :class:`.Scalar` instance into
    a :class:`.ScalarValue` instance"""
    from qnet.algebra.core.scalar_algebra import Scalar, ScalarValue
    scalar_ops = []
    for op in ops:
        if not isinstance(op, Scalar):
            scalar_ops.append(ScalarValue(op))
        else:
            scalar_ops.append(op)
    return scalar_ops, kwargs


def disjunct_hs_zero(cls, ops, kwargs):
    """Return ZeroOperator if all the operators in `ops` have a disjunct
    Hilbert space, or an unchanged `ops`, `kwargs` otherwise
    """
    from qnet.algebra.core.hilbert_space_algebra import TrivialSpace
    from qnet.algebra.core.operator_algebra import ZeroOperator
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
    from qnet.algebra.core.operator_algebra import Commutator
    assert len(ops) == 2
    if cls.order_key(ops[1]) < cls.order_key(ops[0]):
        return -1 * Commutator.create(ops[1], ops[0])
    else:
        return ops, kwargs


def accept_bras(cls, ops, kwargs):
    """Accept operands that are all bras, and turn that into to bra of the
    operation applied to all corresponding kets"""
    from qnet.algebra.core.state_algebra import Bra
    kets = []
    for bra in ops:
        if isinstance(bra, Bra):
            kets.append(bra.ket)
        else:
            return ops, kwargs
    return Bra.create(cls.create(*kets, **kwargs))


def basis_ket_zero_outside_hs(cls, ops, kwargs):
    """For ``BasisKet.create(ind, hs)`` with an integer label `ind`, return a
    :class:`ZeroKet` if `ind` is outside of the range of the underlying Hilbert
    space
    """
    from qnet.algebra.core.state_algebra import ZeroKet
    ind, = ops
    hs = kwargs['hs']
    if isinstance(ind, int):
        if ind < 0 or (hs._dimension is not None and ind >= hs._dimension):
            return ZeroKet
    return ops, kwargs


def indexed_sum_over_const(cls, ops, kwargs):
    r'''Execute an indexed sum over a term that does not depend on the
    summation indices

    .. math::

        \sum_{j=1}^{N} a = N a

    >>> a = symbols('a')
    >>> i, j  = (IdxSym(s) for s in ('i', 'j'))
    >>> unicode(Sum(i, 1, 2)(a))
    '2 a'
    >>> unicode(Sum(j, 1, 2)(Sum(i, 1, 2)(a * i)))
    '∑_{i=1}^{2} 2 i a'
    '''
    term, *ranges = ops
    new_ranges = []
    new_term = term
    for r in ranges:
        if r.index_symbol not in term.free_symbols:
            try:
                new_term *= len(r)
            except TypeError:
                new_ranges.append(r)
        else:
            new_ranges.append(r)
    if len(new_ranges) == 0:
        return new_term
    else:
        return (new_term, ) + tuple(new_ranges), kwargs


def _ranges_key(r, delta_indices):
    """Sorting key for ranges.

    When used with ``reverse=True``, this can be used to sort index ranges into
    the order we would prefer to eliminate them by evaluating KroneckerDeltas:
    First, eliminate primed indices, then indices names higher in the alphabet.
    """
    idx = r.index_symbol
    if idx in delta_indices:
        return (r.index_symbol.primed, r.index_symbol.name)
    else:
        # ranges that are not in delta_indices should remain in the original
        # order
        return (0, ' ')


def indexed_sum_over_kronecker(cls, ops, kwargs):
    """Execute sums over KroneckerDelta prefactors"""
    from qnet.algebra.core.abstract_quantum_algebra import QuantumExpression
    term, *ranges = ops
    assert isinstance(term, QuantumExpression)
    deltas = set(Pattern(head=sympy.KroneckerDelta).findall(term))
    if len(deltas) == 0:
        return ops, kwargs  # nothing to do
    else:  # the term contains at least one KroneckerDelta
        delta_indices = set.union(*[set(
            [idx for idx in delta.free_symbols if isinstance(idx, IdxSym)])
            for delta in deltas])
        ranges = sorted(  # sort in the order we'd prefer to eliminate
            ranges,
            key=partial(_ranges_key, delta_indices=delta_indices),
            reverse=True)
        buffer = [(term, ranges)]
        i = 0  # position in buffer that we're currently handling
        i_range = 0  # position of index-range for current buffer item
        while i < len(buffer):
            t, rs = buffer[i]
            if rs[i_range].index_symbol in delta_indices:
                new_items, flag = _deltasummation(t, rs, i_range)
                new_rs = new_items[0][1]  # same for all new_items
                buffer = buffer[:i] + new_items + buffer[i + 1:]
                assert flag in [1, 2, 3]
                if flag == 2:
                    i_range += 1
                    # * for flag == 1, leaving i_range unchanged will
                    # effectively to to the next range (as the current range
                    # was removed)
                    # * for flag == 3, buffer[i] has changed, and we'll want to
                    # call it again with the same i_range
            else:
                # if the index symbol doesn't occur in any KroneckerDelta,
                # there is no chance _deltasummation will do anything; so we
                # just skip to the next index
                i_range += 1
                new_rs = rs
            if i_range >= len(new_rs):
                # if we've exhausted the index-ranges for the current buffer
                # item, go to the next buffer item
                i += 1
                i_range = 0
        if len(buffer) == 1 and buffer[0] == (term, ranges):
            return ops, kwargs  # couldn't resolve deltas
        else:
            (t, rs) = buffer[0]
            res = t
            if len(rs) > 0:
                res = cls.create(t, *rs, **kwargs)
            for (t, rs) in buffer[1:]:
                if len(rs) > 0:
                    t = cls.create(t, *rs, **kwargs)
                res += t
            return res


def _factors_for_expand_delta(expr):
    """Yield factors from expr, mixing sympy and QNET

    Auxiliary routine for :func:`_expand_delta`.
    """
    from qnet.algebra.core.scalar_algebra import ScalarValue
    from qnet.algebra.core.abstract_quantum_algebra import (
        ScalarTimesQuantumExpression)
    if isinstance(expr, ScalarTimesQuantumExpression):
        yield from _factors_for_expand_delta(expr.coeff)
        yield expr.term
    elif isinstance(expr, ScalarValue):
        yield from _factors_for_expand_delta(expr.val)
    elif isinstance(expr, sympy.Basic) and expr.is_Mul:
        yield from expr.args
    else:
        yield expr


def _expand_delta(expr, idx):
    """Expand the first :class:`sympy.Add` containing a simple
    :class:`sympy.KroneckerDelta`.

    Auxiliary routine for :func:`_deltasummation`. Adapted from SymPy. The
    input `expr` may be a :class:`.QuantumExpression` or a
    `:class:`sympy.Basic` instance.

    Returns a list of summands. The elements of the list may be
    :class:`.QuantumExpression` or a `:class:`sympy.Basic` instances. There is
    no guarantee of type stability: an input :class:`.QuantumExpression` may
    result in a :class:`sympy.Basic` instance in the `summands`.
    """
    found_first_delta = False
    summands = None
    for factor in _factors_for_expand_delta(expr):
        need_to_expand = False
        if not found_first_delta and isinstance(factor, sympy.Basic):
            if factor.is_Add and _has_simple_delta(factor, idx):
                need_to_expand = True
        if need_to_expand:
            found_first_delta = True
            if summands is None:
                summands = list(factor.args)
            else:
                summands = [summands[0]*t for t in factor.args]
        else:
            if summands is None:
                summands = [factor, ]
            else:
                summands = [t*factor for t in summands]
    return summands


def _split_sympy_quantum_factor(expr):
    """Split a product into sympy and qnet factors

    This is a helper routine for applying some sympy transformation on an
    arbitrary product-like expression in QNET. The idea is this::

        expr -> sympy_factor, quantum_factor
        sympy_factor -> sympy_function(sympy_factor)
        expr -> sympy_factor * quantum_factor
    """
    from qnet.algebra.core.abstract_quantum_algebra import (
        QuantumExpression, ScalarTimesQuantumExpression)
    from qnet.algebra.core.scalar_algebra import ScalarValue, ScalarTimes, One
    if isinstance(expr, ScalarTimesQuantumExpression):
        sympy_factor, quantum_factor = _split_sympy_quantum_factor(expr.coeff)
        quantum_factor *= expr.term
    elif isinstance(expr, ScalarValue):
        sympy_factor = expr.val
        quantum_factor = expr._one
    elif isinstance(expr, ScalarTimes):
        sympy_factor = sympy.S(1)
        quantum_factor = expr._one
        for op in expr.operands:
            op_sympy, op_quantum = _split_sympy_quantum_factor(op)
            sympy_factor *= op_sympy
            quantum_factor *= op_quantum
    elif isinstance(expr, sympy.Basic):
        sympy_factor = expr
        quantum_factor = One
    else:
        sympy_factor = sympy.S(1)
        quantum_factor = expr
    assert isinstance(sympy_factor, sympy.Basic)
    assert isinstance(quantum_factor, QuantumExpression)
    return sympy_factor, quantum_factor


def _extract_delta(expr, idx):
    """Extract a "simple" Kronecker delta containing `idx` from `expr`.

    Assuming `expr` can be written as the product of a Kronecker Delta and a
    `new_expr`, return a tuple of the sympy.KroneckerDelta instance and
    `new_expr`. Otherwise, return a tuple of None and the original `expr`
    (possibly converted to a :class:`.QuantumExpression`).

    On input, `expr` can be a :class:`QuantumExpression` or a
    :class:`sympy.Basic` object. On output, `new_expr` is guaranteed to be a
    :class:`QuantumExpression`.
    """
    from qnet.algebra.core.abstract_quantum_algebra import QuantumExpression
    from qnet.algebra.core.scalar_algebra import ScalarValue
    sympy_factor, quantum_factor = _split_sympy_quantum_factor(expr)
    delta, new_expr = _sympy_extract_delta(sympy_factor, idx)
    if delta is None:
        new_expr = expr
    else:
        new_expr = new_expr * quantum_factor
    if isinstance(new_expr, ScalarValue._val_types):
        new_expr = ScalarValue.create(new_expr)
    assert isinstance(new_expr, QuantumExpression)
    return delta, new_expr


def _deltasummation(term, ranges, i_range):
    """Partially execute a summation for `term` with a Kronecker Delta for one
    of the summation indices.

    This implements the solution to the core sub-problem in
    :func:`indexed_sum_over_kronecker`

    Args:
        term (QuantumExpression): term of the sum
        ranges (list): list of all summation index ranges
            (class:`IndexRangeBase` instances)
        i_range (int): list-index of element in `ranges` which should be
            eliminated

    Returns:
        ``(result, flag)`` where `result` is a list
        of ``(new_term, new_ranges)`` tuples and `flag` is an integer.

    There are three possible cases, indicated by the returned `flag`. Consider
    the following setup::

        >>> i, j, k = symbols('i, j, k', cls=IdxSym)
        >>> i_range = IndexOverList(i, (0, 1))
        >>> j_range = IndexOverList(j, (0, 1))
        >>> ranges = [i_range, j_range]
        >>> def A(i, j):
        ...    from sympy import IndexedBase
        ...    return OperatorSymbol(StrLabel(IndexedBase('A')[i, j]), hs=0)

    1. If executing the sum produces a single non-zero term, result will be
    ``[(new_term, new_ranges)]`` where `new_ranges` contains the input `ranges`
    without the eliminated range specified by `i_range`.  This should be the
    most common case for calls to:func:`_deltasummation`::

        >>> term = KroneckerDelta(i, j) * A(i, j)
        >>> result, flag = _deltasummation(term, [i_range, j_range], 1)
        >>> assert result == [(A(i, i), [i_range])]
        >>> assert flag == 1

    2. If executing the sum for the index symbol specified via `index_range`
    does not reduce the sum, the result will be the list ``[(term, ranges)]``
    with unchanged `term` and `ranges`::

        >>> term = KroneckerDelta(j, k) * A(i, j)
        >>> result, flag = _deltasummation(term, [i_range, j_range], 0)
        >>> assert result == [(term, [i_range, j_range])]
        >>> assert flag == 2

    This case also covers if there is no Kroncker delta in the term::

        >>> term = A(i, j)
        >>> result, flag = _deltasummation(term, [i_range, j_range], 0)
        >>> assert result == [(term, [i_range, j_range])]
        >>> assert flag == 2

    3. If `term` does not contain a Kronecker delta as a factor, but in a
    sum that can be expanded, the result will be a list of
    ``[(summand1, ranges), (summand2, ranges), ...]`` for the summands of that
    expansion. In this case, `:func:`_deltasummation` should be called again
    for every tuple in the list, with the same `i_range`::

        >>> term = (KroneckerDelta(i, j) + 1) * A(i, j)
        >>> result, flag = _deltasummation(term, [i_range, j_range], 1)
        >>> assert result == [
        ...     (A(i, j), [i_range, j_range]),
        ...     (KroneckerDelta(i,j) * A(i, j), [i_range, j_range])]
        >>> assert flag == 3
    """
    from qnet.algebra.core.abstract_quantum_algebra import QuantumExpression
    idx = ranges[i_range].index_symbol
    summands = _expand_delta(term, idx)
    if len(summands) > 1:
        return [(summand, ranges) for summand in summands], 3
    else:
        delta, expr = _extract_delta(summands[0], idx)
    if not delta:
        return [(term, ranges)], 2
    solns = sympy.solve(delta.args[0] - delta.args[1], idx)
    assert len(solns) > 0  # I can't think of an example that might cause this
    #     if len(solns) == 0:
    #         return [(term._zero, [])], 4
    if len(solns) != 1:
        return [(term, ranges)], 2
    value = solns[0]
    new_term = expr.substitute({idx: value})
    if _RESOLVE_KRONECKER_WITH_PIECEWISE:
        new_term *= ranges[i_range].piecewise_one(value)
    assert isinstance(new_term, QuantumExpression)
    return [(new_term, ranges[:i_range] + ranges[i_range+1:])], 1


def derivative_via_diff(cls, ops, kwargs):
    """Implementation of the :meth:`QuantumDerivative.create` interface via the
    use of :meth:`QuantumExpression._diff`.

    Thus, by having :meth:`.QuantumExpression.diff` delegate to
    :meth:`.QuantumDerivative.create`, instead of
    :meth:`.QuantumExpression._diff` directly, we get automatic caching of
    derivatives
    """
    assert len(ops) == 1
    op = ops[0]
    derivs = kwargs['derivs']
    vals = kwargs['vals']
    # both `derivs` and `vals` are guaranteed to be tuples, via the conversion
    # that's happening in `QuantumDerivative.create`
    for (sym, n) in derivs:
        if sym.free_symbols.issubset(op.free_symbols):
            for k in range(n):
                op = op._diff(sym)
        else:
            return op.__class__._zero
    if vals is not None:
        try:
            # for QuantumDerivative instance
            return op.evaluate_at(vals)
        except AttributeError:
            # for explicit Expression
            return op.substitute(vals)
    else:
        return op
