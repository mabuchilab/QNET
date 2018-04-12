from sympy import symbols
import unittest
from collections import OrderedDict
import pytest

from qnet.algebra.abstract_algebra import (
    Operation, assoc, assoc_indexed, indexed_sum_over_const, orderby,
    filter_neutral, CannotSimplify, match_replace_binary, idem, IndexedSum,
    ScalarTimesExpression)
from qnet.algebra.indices import IndexOverRange, IdxSym
from qnet.algebra.ordering import expr_order_key
from qnet.algebra.pattern_matching import pattern_head, wc
from qnet.algebra.operator_algebra import (
    LocalSigma, LocalProjector, OperatorTimes, Displace)
from qnet.algebra.hilbert_space_algebra import LocalSpace


class TestOperationSimplifcations(unittest.TestCase):

    def setUp(self):

        class Flat(Operation):
            _simplifications = [assoc, ]

        class Orderless(Operation):
            order_key = expr_order_key
            _simplifications = [orderby, ]

        class FilterNeutral(Operation):
            neutral_element = object()
            _simplifications = [filter_neutral, ]

        def mult_str_int_if_pos(s, i):
            if i >= 0:
                return s * i
            raise CannotSimplify()

        def mult_inv_str_int_if_neg(s, i):
            return s[::-1]*(-i)

        a_int = wc("a", head=int)
        a_negint = wc("a", head=int, conditions=[lambda a: a < 0, ])
        a_str = wc("a", head=str)
        b_str = wc("b", head=str)

        class MatchReplaceBinary(Operation):
            _binary_rules = OrderedDict([
                ('r1', (pattern_head(a_int, b_str),
                    lambda a, b: mult_str_int_if_pos(b,a))),
                ('r2', (pattern_head(a_negint, b_str),
                    lambda a, b: mult_inv_str_int_if_neg(b,a))),
                ('r3', (pattern_head(a_str, b_str),
                    lambda a, b: a + b))
            ])
            neutral_element = 1
            _simplifications = [assoc, match_replace_binary]

        class Idem(Operation):
            order_key = expr_order_key
            _simplifications = [assoc, idem]

        class AssocIndexed(IndexedSum):
            _simplifications = [assoc_indexed]

            def __mul__(self, other):
                return ScalarTimesExpression.create(other, self)

            def __rmul__(self, other):
                return ScalarTimesExpression.create(other, self)

        class AssocIndexed2(AssocIndexed):
            _simplifications = [indexed_sum_over_const]

        self.Flat = Flat
        self.Orderless = Orderless
        self.FilterNeutral = FilterNeutral
        self.MatchReplaceBinary = MatchReplaceBinary
        self.Idem = Idem
        self.AssocIndexed = AssocIndexed
        self.AssocIndexed2 = AssocIndexed2


    def testFlat(self):
        assert self.Flat.create(1,2,3, self.Flat(4,5,6),7,8) == \
                         self.Flat(1,2,3,4,5,6,7,8)


    def testOrderless(self):
        assert self.Orderless.create(3,1,2) == self.Orderless(1,2,3)

    def testFilterNeutral(self):
        one = self.FilterNeutral.neutral_element
        assert self.FilterNeutral.create(1,2,3,one,4,5,one) == \
                         self.FilterNeutral(1,2,3,4,5)
        assert self.FilterNeutral.create(one) == one

    def testSimplifyBinary(self):
        assert self.MatchReplaceBinary.create(1,2,"hallo") == "hallohallo"
        assert self.MatchReplaceBinary.create(-1,"hallo") == "ollah"
        assert self.MatchReplaceBinary.create(-3,"hallo") == "ollahollahollah"
        assert self.MatchReplaceBinary.create(2,-2,"hallo") == \
                         "ollahollahollahollah"
        assert self.MatchReplaceBinary.create("1","2","3") == "123"

    def testIdem(self):
        assert self.Idem.create(2,3,3,1,2,3,4,1,2) == \
                         self.Idem(1,2,3,4)

    def testAssocIndexed(self):

        def r(index_symbol):
            i = index_symbol
            if not isinstance(i, IdxSym):
                i = IdxSym(i)
            return IndexOverRange(i, 0, 2)

        def sum(term, *indices):
            return self.AssocIndexed(term, *[r(i) for i in indices])

        a = symbols('a')

        expr = self.AssocIndexed.create(
            sum("term", 'i'), r('j'))
        assert expr == sum("term", 'j', 'i')

        expr = self.AssocIndexed.create(
            sum("term", 'i'), r('i'))
        assert expr == sum("term", 'i', IdxSym('i', primed=1))

        expr = self.AssocIndexed.create(
            a * sum("term", 'i'), r('j'))
        assert expr == a * sum("term", 'j', 'i')

        expr = self.AssocIndexed.create(
            IdxSym('j') * sum(a, 'i'), r('j'))
        assert expr == sum(IdxSym('j') * a, 'j', 'i')

    def testIndexedSumOverConst(self):
        a = symbols('a')
        expr = self.AssocIndexed2.create(
            a, IndexOverRange(IdxSym('i'), 0, 2))
        assert expr == 3 * a



def test_match_replace_binary_complete():
    """Test that replace_binary works correctly for a non-trivial case"""
    x, y, z, alpha = symbols('x y z alpha')
    hs = LocalSpace('f')
    ops = [LocalProjector(0, hs=hs),
           Displace(-alpha, hs=hs),
           Displace(alpha, hs=hs),
           LocalProjector(0, hs=hs)]
    res = OperatorTimes.create(*ops)
    assert res == LocalProjector(0, hs=hs)
