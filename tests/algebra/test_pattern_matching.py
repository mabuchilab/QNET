from collections import OrderedDict

from sympy import Symbol
import pytest

from qnet.algebra.core.scalar_algebra import Scalar
from qnet.algebra.core.scalar_algebra import ScalarValue
from qnet.algebra.core.operator_algebra import (
        OperatorSymbol, ScalarTimesOperator, OperatorTimes, Operator,
        LocalOperator, LocalSigma)
from qnet.algebra.library.fock_operators import Create
from qnet.algebra.core.hilbert_space_algebra import (
    FullSpace, HilbertSpace, LocalSpace)
from qnet.algebra.core.circuit_algebra import (
        Circuit, CPermutation, Concatenation, SeriesProduct, CircuitSymbol,
        Feedback)
from qnet.algebra.pattern_matching import (
        wc, Pattern, pattern, pattern_head, MatchDict, ProtoExpr,
        match_pattern)


def test_match_dict():
    """Test the the behavior of MatchDict as a write-once dictionary"""
    d = MatchDict(OrderedDict([(1, 2), ('a', 1)]))
    assert d['a'] == 1
    d['b'] = 5
    assert d['b'] == 5
    d['b'] = 5
    assert d['b'] == 5
    assert list(d.keys()) == [1, 'a', 'b']
    assert list(d.values()) == [2, 1, 5]
    assert list(d.items()) == [(1, 2), ('a', 1), ('b', 5)]
    with pytest.raises(KeyError) as exc_info:
        d['b'] = 4
    assert 'has already been set' in str(exc_info)
    assert d['b'] == 5
    d2 = MatchDict({3: 1, 4: 2})
    d.update(d2)
    assert d == {1: 2, 'a': 1, 'b': 5, 3: 1, 4: 2}
    assert d.success
    d2.success = False
    d2.reason = "Test of updating with non-match"
    d.update(d2)
    assert not d.success
    assert d.reason == "Test of updating with non-match"
    d.update({6: 1, 7: 1})  # updating with a regular dict, not MatchDict
    assert d[7] == 1
    with pytest.raises(KeyError) as exc_info:
        d.update({5: 2, 3: 2, 4: 3})
    assert 'has already been set' in str(exc_info)
    with pytest.raises(KeyError) as exc_info:
        del d[5]
    assert 'Read-only dictionary' in str(exc_info)
    d['t'] = [1, ]
    d['t'].append(2)
    assert d['t'] == [1, 2]
    d['t'] = [1, 2]
    with pytest.raises(KeyError) as exc_info:
        d['t'] = [3, 4]
    assert 'has already been set' in str(exc_info)
    assert d.merge_lists == 0
    d.merge_lists = -1
    d['t'] = [3, 4]
    assert d['t'] == [1, 2, 3, 4]
    d.merge_lists = +1
    d['t'] = [3, 4]
    assert d['t'] == [3, 4, 1, 2, 3, 4]
    d['t'] = d['t']
    assert d['t'] == [3, 4, 1, 2, 3, 4, 3, 4, 1, 2, 3, 4]


def test_proto_expr_as_sequence():
    """Test sequence interface of proto-expressions"""
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    proto_expr = ProtoExpr.from_expr(a)
    assert len(proto_expr) == 2
    assert proto_expr[0] == 'a'
    assert proto_expr[1] == h1


def test_wc():
    """Test that the wc() constructor produces the equivalent Pattern
    instance"""
    patterns = [
        (wc(),
         Pattern(head=None, args=None, kwargs=None, mode=Pattern.single,
                 wc_name=None, conditions=None)
         ),
        (wc('a'),
         Pattern(head=None, args=None, kwargs=None, mode=Pattern.single,
                 wc_name='a', conditions=None)
        ),
        (wc('a_'), wc('a')),
        (wc('a__'),
         Pattern(head=None, args=None, kwargs=None, mode=Pattern.one_or_more,
                 wc_name='a', conditions=None)
         ),
        (wc('a___'),
         Pattern(head=None, args=None, kwargs=None, mode=Pattern.zero_or_more,
                 wc_name='a', conditions=None)
         ),
        (wc('a', head=int),
         Pattern(head=int, args=None, kwargs=None, mode=Pattern.single,
                 wc_name='a', conditions=None)
         ),
        (wc('a', head=(int, float)),
         Pattern(head=(int, float), args=None, kwargs=None,
                 mode=Pattern.single, wc_name='a', conditions=None)
         ),
    ]
    for pat1, pat2 in patterns:
        print(repr(pat1))
        assert pat1 == pat2
    with pytest.raises(ValueError):
        wc("____")


def test_pattern():
    """Test that the pattern() constructor produces the equivalent Pattern
    instance"""
    true_cond = lambda expr: True
    patterns = [
        (pattern(OperatorSymbol, 'O', hs=FullSpace),
         Pattern(head=OperatorSymbol, args=['O', ], kwargs={'hs': FullSpace})
         ),
        (pattern(OperatorSymbol, 'O', a=1, b=2, hs=FullSpace),
         Pattern(head=OperatorSymbol, args=['O', ],
                 kwargs={'a': 1, 'b': 2, 'hs': FullSpace})
         ),
        (pattern(OperatorSymbol, 'O', a=1, b=2, hs=FullSpace,
                 conditions=[true_cond, ]),
         Pattern(head=OperatorSymbol, args=['O', ],
                 kwargs={'a': 1, 'b': 2, 'hs': FullSpace},
                 conditions=[true_cond, ])
         ),
    ]
    for pat1, pat2 in patterns:
        print(repr(pat1))
        assert pat1 == pat2


def test_invalid_pattern():
    """Test that instantiating a Pattern with invalid attributes raises the
    appropriate exceptions"""
    with pytest.raises(TypeError) as exc_info:
        Pattern(head='OperatorSymbol')
    assert 'must be class' in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        pattern(ScalarTimesOperator, wc('a'), wc('b__'), wc('c'))
    assert ('Only the first or last argument may have a mode indicating an '
            'occurrence of more than 1' in str(exc_info))
    with pytest.raises(ValueError) as exc_info:
        wc('a_____')
    assert "Invalid name_mode" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        pattern(ScalarTimesOperator, wc('a'), wc('b'), wc_name='S', mode=5)
    assert "Mode must be one of" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        pattern(ScalarTimesOperator, wc('a'), wc('b'), wc_name='S', mode='1')
    assert "Mode must be one of" in str(exc_info)



def test_pattern_head():
    """Test that the pattern_head() constructor produces the equivalent Pattern
    instance"""
    true_cond = lambda expr: True
    patterns = [
        (pattern_head('O', FullSpace),
         Pattern(args=['O', FullSpace], kwargs=None)
         ),
        (pattern_head('O', FullSpace, a=1, b=2),
         Pattern(args=['O', FullSpace], kwargs={'a': 1, 'b': 2})
         ),
        (pattern_head('O', FullSpace, a=1, b=2, conditions=[true_cond, ]),
         Pattern(args=['O', FullSpace], kwargs={'a': 1, 'b': 2},
                 conditions=[true_cond, ])
         ),
    ]
    for pat1, pat2 in patterns:
        print(repr(pat1))
        assert pat1 == pat2


# test expressions
two_t = 2 * Symbol('t')
two_O = 2 * OperatorSymbol('O', hs=FullSpace)
proto_two_O = ProtoExpr([2, OperatorSymbol('O', hs=FullSpace)], {})
proto_kwargs = ProtoExpr([1, 2], {'a': '3', 'b': 4})
proto_kw_only = ProtoExpr([], {'a': 1, 'b': 2})
proto_ints2 = ProtoExpr([1, 2], {})
proto_ints3 = ProtoExpr([1, 2, 3], {})
proto_ints4 = ProtoExpr([1, 2, 3, 4], {})
proto_ints5 = ProtoExpr([1, 2, 3, 4, 5], {})
C1 = CircuitSymbol('C1', cdim=3)
C2 = CircuitSymbol('C2', cdim=3)
C3 = CircuitSymbol('C3', cdim=3)
C4 = CircuitSymbol('C4', cdim=3)
perm1 = CPermutation((2, 1, 0))
perm2 = CPermutation((0, 2, 1))
concat_expr = Concatenation(SeriesProduct(C1, C2, perm1),
                            SeriesProduct(C3, C4, perm2))
concat_expr2 = Concatenation(SeriesProduct(perm1, C1, C2),
                             SeriesProduct(perm2, C3, C4))
expr_fb = Feedback(C1, out_port=1, in_port=2)

# test patterns and wildcards
wc_a_int_2 = wc('a', head=(ScalarValue, int), conditions=[lambda i: i == 2, ])
wc_a_int_3 = wc('a', head=(ScalarValue, int), conditions=[lambda i: i == 3, ])
wc_a_int = wc('a', head=int)
wc_label_str = wc('label', head=str)
wc_hs = wc('space', head=HilbertSpace)
pattern_two_O = pattern(ScalarTimesOperator,
                        wc_a_int_2,
                        pattern(OperatorSymbol, wc_label_str, hs=wc_hs))
pattern_two_O_head = pattern_head(wc_a_int_2,
                                  pattern(OperatorSymbol, wc_label_str,
                                          hs=wc_hs))
pattern_two_O_expr = pattern(ScalarTimesOperator,
                             wc_a_int_2, OperatorSymbol('O', hs=FullSpace))
pattern_kwargs = pattern_head(wc('i1', head=int), wc('i2', head=int),
                              a=wc('a', head=str), b=wc('b', head=int))
pattern_kw_only = pattern_head(a=pattern(int), b=pattern(int))

conditions = [lambda c: c.cdim == 3, lambda c: c.label[0] == 'C']
A__Circuit = wc("A__", head=CircuitSymbol, conditions=conditions)
C__Circuit = wc("C__", head=CircuitSymbol, conditions=conditions)
B_CPermutation = wc("B", head=CPermutation)
D_CPermutation = wc("D", head=CPermutation)
pattern_concat = pattern(
        Concatenation,
        pattern(SeriesProduct, A__Circuit, B_CPermutation),
        pattern(SeriesProduct, C__Circuit, D_CPermutation))
pattern_concat2 = pattern(
        Concatenation,
        pattern(SeriesProduct, B_CPermutation, A__Circuit),
        pattern(SeriesProduct, D_CPermutation, C__Circuit))
A_Circuit = wc("A", head=Circuit)
pattern_ApA = pattern(Concatenation, A_Circuit, A_Circuit)

pattern_ints = pattern_head(pattern(int), pattern(int), pattern(int),
                            wc('i___', head=int))
pattern_ints5 = pattern_head(1, 2, 3, 4, 5)
pattern_fb = wc('B', head=Feedback,
                args=[A_Circuit, ],
                kwargs={'out_port': pattern(int), 'in_port': pattern(int)},
                )

SCALAR_TYPES = Scalar._val_types

PATTERNS = [
#   (ind pattern,               expr,      matched?,  wc_dict)
    (1,  wc(),                  1,             True,  {}),
    (2,  wc('i__', head=int),   1,             True,  {'i': [1, ]}),
    (3,  wc(),                  two_t,         True,  {}),
    (4,  wc(),                  two_O,         True,  {}),
    (5,  wc('a'),               two_t,         True,  {'a': two_t}),
    (6,  wc('a'),               two_O,         True,  {'a': two_O}),
    (7,  pattern(SCALAR_TYPES), two_t,         True,  {}),
    (8,  pattern(SCALAR_TYPES), two_O,         False, {}),
    (9,  pattern_two_O,         two_O,         True,  {'a': 2, 'label': 'O',
                                                       'space': FullSpace}),
    (10, pattern_two_O_head,    proto_two_O,   True,  {'a': 2, 'label': 'O',
                                                       'space': FullSpace}),
    (11, pattern_two_O_expr,    two_O,         True,  {'a': 2}),
    (12, pattern_two_O,         two_t,         False, {}),
    (13, pattern_kwargs,        proto_kwargs,  True,  {'i1': 1, 'i2': 2,
                                                       'a': '3', 'b': 4}),
    (14, pattern_kw_only,       proto_kw_only, True,  {}),
    (15, pattern_ints,          proto_ints2,   False, {}),
    (16, pattern_ints,          proto_ints3,   True,  {'i': []}),
    (17, pattern_ints,          proto_ints4,   True,  {'i': [4, ]}),
    (18, pattern_ints,          proto_ints5,   True,  {'i': [4, 5]}),
    (19, pattern_ints5,         proto_ints5,   True,  {}),
    (20, pattern_concat,        concat_expr,   True,  {
                        'A': [C1, C2], 'B': perm1, 'C': [C3, C4], 'D': perm2}),
    (21, pattern_concat,        concat_expr2,  False, {}),
    (22, pattern_concat2,       concat_expr,   False, {}),
    (23, pattern_concat2,       concat_expr2,  True,  {
                        'A': [C1, C2], 'B': perm1, 'C': [C3, C4], 'D': perm2}),
    (24, pattern_ApA,           C1+C1,         True,  {'A': C1}),
    (25, pattern_fb,            expr_fb,       True,  {'A': C1, 'B': expr_fb}),
]


@pytest.mark.parametrize('ind, pat, expr, matched, wc_dict', PATTERNS)
def test_match(ind, pat, expr, matched, wc_dict):
    """Test that patterns match expected expressions and produce the correct
    match dict """
    # `ind` is just so that we can track *which* rule fails, is there is a
    # failure
    print("%s.match(%s)" % (repr(pat), repr(expr)))
    match = pat.match(expr)
    assert bool(match) == matched
    if matched:
        assert len(match) == len(wc_dict)
        print("   -> %s" % str(match))
        for key, val in wc_dict.items():
            assert match[key] == val
    else:
        print("   -> NO MATCH (%s)" % match.reason)


def test_no_match():
    """Test that matches fail for the correct reason"""

    conds = [lambda i: i > 0, lambda i: i < 10]
    match = wc('i__', head=int, conditions=conds).match(10)
    assert not match
    assert 'does not meet condition 2' in match.reason

    pat = pattern_head(pattern(int), pattern(int), wc('i___', head=int))
    match = pat.match(ProtoExpr([1, ], {}))
    assert not match
    assert 'insufficient number of arguments' in match.reason

    pat = pattern_head(1, 2, 3)
    match = pat.match(ProtoExpr([1, 2], {}))
    assert not match
    assert 'insufficient number of arguments' in match.reason

    pat = pattern_head(pattern(int), wc('i__', head=int))
    match = pat.match(ProtoExpr([1, ], {}))
    assert not match
    assert 'insufficient number of arguments' in match.reason

    pat = pattern_head(a=pattern(int), b=pattern(int))
    match = pat.match(ProtoExpr([], {'a': 1, 'c': 2}))
    assert not match
    assert "has no keyword argument 'b'" in match.reason

    pat = pattern_head(a=pattern(int), b=pattern(str))
    match = pat.match(ProtoExpr([], {'a': 1, 'b': 2}))
    assert not match
    assert "2 is not an instance of str" in match.reason

    pat = pattern_head(a=pattern(int),
                       b=pattern_head(pattern(int), pattern(int)))
    match = pat.match(ProtoExpr([], {'a': 1, 'b': 2}))
    assert not match
    assert "2 is not an instance of ProtoExpr" in match.reason

    pat = pattern_head(pattern(int))
    match = pat.match(ProtoExpr([1, 2], {}))
    assert not match
    assert 'too many positional arguments' in match.reason

    match = pattern_ApA.match(C1 + C2)
    assert not match
    assert "Double wildcard: 'A has already been set'" in match.reason

    pat = wc('A', head=Feedback, args=[A_Circuit, ],
             kwargs={'out_port': 1, 'in_port': 2})
    match = pat.match(Feedback(C1, out_port=1, in_port=2))
    assert not match
    assert "Double wildcard: 'A has already been set'" in match.reason

    match = match_pattern(1, 2)
    assert not match.success
    assert "Expressions '1' and '2' are not the same" in match.reason


def test_pattern_str():
    assert str(pattern_kwargs) == (
        "Pattern(head=ProtoExpr, args=(Pattern(head=int, wc_name='i1'), "
        "Pattern(head=int, wc_name='i2')), kwargs={'a': Pattern(head=str, "
        "wc_name='a'), 'b': Pattern(head=int, wc_name='b')})")


def test_findall():
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    c = OperatorSymbol("c", hs=h1)
    h1_custom = LocalSpace("h1", local_identifiers={'Create': 'c'})
    c_local = Create(hs=h1_custom)

    expr = 2 * (a * b * c - b * c * a + a * b)
    op_symbols = pattern(OperatorSymbol).findall(expr)
    assert len(op_symbols) == 8
    assert set(op_symbols) == {a, b, c}
    op = wc(head=Operator)
    three_factors = pattern(OperatorTimes, op, op, op).findall(expr)
    assert three_factors == [a * b * c, b * c * a]
    assert len(pattern(LocalOperator).findall(expr)) == 0
    assert len(pattern(LocalOperator)
               .findall(expr.substitute({c: c_local}))) == 2


def test_finditer():
    h1 = LocalSpace("h1")
    a = OperatorSymbol("a", hs=h1)
    b = OperatorSymbol("b", hs=h1)
    c = OperatorSymbol("c", hs=h1)
    h1_custom = LocalSpace("h1", local_identifiers={'Create': 'c'})
    c_local = Create(hs=h1_custom)

    expr = 2 * (a * b * c - b * c * a + a * b)
    pat = wc('sym', head=OperatorSymbol)
    for m in pat.finditer(expr):
        assert 'sym' in m
    matches = list(pat.finditer(expr))
    assert len(matches) == 8
    op_symbols = [m['sym'] for m in matches]
    assert set(op_symbols) == {a, b, c}

    op = wc(head=Operator)
    three_factors = pattern(OperatorTimes, op, op, op).findall(expr)
    assert three_factors == [a * b * c, b * c * a]
    assert len(list(pattern(LocalOperator).finditer(expr))) == 0
    assert len(list(pattern(LocalOperator)
                    .finditer(expr.substitute({c: c_local})))) == 2


def test_wc_names():
    """Test the wc_names property"""
    ra = wc("ra", head=(int, str))
    rb = wc("rb", head=(int, str))
    rc = wc("rc", head=(int, str))
    rd = wc("rd", head=(int, str))
    ls = wc("ls", head=LocalSpace)
    pat = pattern_head(
        pattern(LocalSigma, ra, rb, hs=ls),
        pattern(LocalSigma, rc, rd, hs=ls))
    assert pat.wc_names == set(['ra', 'rb', 'rc', 'rd', 'ls'])
