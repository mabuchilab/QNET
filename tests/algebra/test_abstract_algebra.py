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

from qnet.algebra.abstract_algebra import *
import unittest


class _Dummy1(Operation):
    pass


class _Dummy2(Operation):
    pass



class TestPatternMatching(unittest.TestCase):


    def testWC(self):
        self.assertEqual(wc(),Wildcard())
        self.assertEqual(wc("a"),Wildcard("a"))
        self.assertEqual(wc("a_"), Wildcard("a", mode = 1))
        self.assertEqual(wc("a__"), Wildcard("a", mode = 2))
        self.assertEqual(wc("__"), Wildcard(mode = 2))
        self.assertEqual(wc("a___"), Wildcard("a", mode = 3))
        self.assertRaises(ValueError, wc, "____")

    def testMatchRange(self):
        self.assertEqual(match_range(wc("a")), (1,1))
        self.assertEqual(match_range(wc("a__")), (1,inf))
        self.assertEqual(match_range(wc("a___")), (0,inf))
        self.assertEqual(match_range(PatternTuple((wc("a"),1,2))),(3,3))
        self.assertEqual(match_range(PatternTuple((wc("a"),1,2,wc("b__")))),(4,inf))
        self.assertEqual(match_range(PatternTuple((wc("a"),1,2,wc("b___")))),(3,inf))


    def testUpdatePattern(self):
        a = wc("a_")
        b = wc("b__")
        c = wc("c___")
        self.assertEqual(update_pattern(a, Match(a = 1)), 1)
        self.assertEqual(update_pattern(b, Match(b = 1)), 1)
        self.assertEqual(update_pattern(c, Match(c = 1)), 1)
        self.assertEqual(update_pattern(b, Match(b = OperandsTuple((1,)))), (1,))
        self.assertEqual(update_pattern(c, Match(c = OperandsTuple((1,)))), (1,))
        self.assertEqual(update_pattern(PatternTuple((a,)), Match(a = 1)), (1,))
        self.assertEqual(update_pattern(PatternTuple((b,)), Match(b = OperandsTuple((1,)))), (1,))
        self.assertEqual(update_pattern(PatternTuple((1,b,b)), Match(b = OperandsTuple((1,2)))), (1,1,2,1,2))
        self.assertEqual(update_pattern(PatternTuple((b,a)), Match(b = OperandsTuple((1,)))), (1,a))

    def testFlatWithWildcard(self):

        a = wc("a", head = int)
        b = wc("b")

        self.assertEqual(match(a, _Dummy1(1,3,2)), False)
        self.assertEqual(match(b, _Dummy1(1,3,2)), Match(b =  _Dummy1(1,3,2)))
        self.assertEqual(match(_Dummy1(1,3,2), _Dummy1(1,3,2)), Match())


    def testNestedWithWildcards(self):
        a = wc("a", head = int)
        b = wc("b")
        c = wc("c", head = _Dummy2)
        self.assertEqual(match(PatternTuple((b,b)), OperandsTuple((("hallo",[]), ("hallo",[])))), Match(b = ("hallo",[])))
        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo",2)), Match(b = "hallo"))
        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,3,2)), Match(a = 3))
        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1,"hallo","hallo")), Match(b =  "hallo"))
        self.assertEqual(match(_Dummy1(_Dummy2(a,1),b,2), _Dummy1(_Dummy2(1,1),3,2)), Match(a=1, b=3))
        self.assertEqual(match(_Dummy1(_Dummy2(a,1),a,2), _Dummy1(_Dummy2(1,1),1,2)), Match(a=1))
        self.assertEqual(match(_Dummy1(_Dummy2(a,1),a,2), _Dummy1(_Dummy2(1,1),-1,2)), False)
        self.assertEqual(match(_Dummy1(c,2), _Dummy1(_Dummy2(1,2,3),2)), Match(c=_Dummy2(1,2,3)))


    def testOneOrMore(self):
        a = wc("a__", head = int)
        b = wc("b__")
        self.assertEqual(match(b, OperandsTuple((1,2,3))), Match(b=(1,2,3)))
        self.assertEqual(match(a, OperandsTuple((1,2,3))), Match(a=(1,2,3)))
        self.assertEqual(match(b, _Dummy1(1,3,4,2)), Match(b=(_Dummy1(1,3,4,2),)))
        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,3,4,2)), Match(a=(3,4)))
        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,3,2)), Match(a= (3,)))

        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo",2)), Match(b=("hallo",)))
        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo",1,2)), Match(b= ("hallo",1)))
        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo","du",2)), Match(b=("hallo","du")))

        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,"hallo","du",2)), False)
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1,"hallo","hallo")), Match(b= ("hallo",)))

    def testZeroOrMore(self):
        a = wc("a___", head = int)
        b = wc("b___")

        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,2)), Match(a=()))
        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,3,2)), Match(a=(3,)))

        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,2)), Match(b=()))
        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo",2)), Match(b=("hallo",)))
        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo",1,2)), Match(b=("hallo",1)))
        self.assertEqual(match(_Dummy1(1,b,2), _Dummy1(1,"hallo","du",2)), Match(b=("hallo","du")))

        self.assertEqual(match(_Dummy1(1,a,2), _Dummy1(1,"hallo","du",2)), False)
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1)), Match(b=()))
        self.assertEqual(match(_Dummy1(1,b,b), _Dummy1(1,"hallo","hallo")), Match(b= ("hallo",)))


class TestPatternSubstitution(unittest.TestCase):
    def testFlatNoWildcards(self):
        pass

    def testNestedNoWildcards(self):
        pass

    def testFlatWildcards(self):
        pass

    def testNestedWildcards(self):
        pass


class TestOperationDecorators(unittest.TestCase):

    def setUp(self):

        @assoc
        class Flat(Operation):
            pass


        @orderby
        class Orderless(Operation):
            pass


        @filter_neutral
        class FilterNeutral(Operation):
            neutral_element = object()


        @check_signature
        class CheckSignature(Operation):
            signature = (int, long, float, complex), str

        @assoc
        @check_signature_assoc
        class CheckSignatureFlat(Operation):
            signature = (int, long, float, complex),


        def mult_str_int_if_pos(s, i):
            if i >= 0:
                return s * i
            raise CannotSimplify()

        def mult_inv_str_int_if_neg(s, i):
            return s[::-1]*(-i)

        a_int = wc("a",head = int)
        a_negint = wc("a", head = int, condition = lambda a: a < 0)
        b_int = wc("b",head = int)
        c_str = wc("c",head = str)

        @assoc
        @match_replace_binary
        class MatchReplaceBinary(Operation):
            _binary_rules = [
                ((a_int, b_int), lambda a,b: a + b),
                ((a_int, c_str), lambda c,a: mult_str_int_if_pos(c,a)),
                ((a_negint, c_str), lambda c,a: mult_inv_str_int_if_neg(c,a))
            ]


        @assoc
        @idem
        class Idem(Operation):
            pass



        self.Flat = Flat
        self.Orderless = Orderless
        self.FilterNeutral = FilterNeutral
        self.CheckSignature = CheckSignature
        self.CheckSignatureFlat = CheckSignatureFlat
        self.MatchReplaceBinary = MatchReplaceBinary
        self.Idem = Idem



    def testFlat(self):
        self.assertEqual(self.Flat.create(1,2,3, self.Flat(4,5,6),7,8), self.Flat(1,2,3,4,5,6,7,8))



    def testOrderless(self):
        self.assertEqual(self.Orderless.create(3,1,2), self.Orderless(1,2,3))


    def testFilterNeutral(self):
        one = self.FilterNeutral.neutral_element
        self.assertEqual(self.FilterNeutral.create(1,2,3,one,4,5,one), self.FilterNeutral(1,2,3,4,5))
        self.assertEqual(self.FilterNeutral.create(one), one)


    def testSimplifyBinary(self):
        self.assertEqual(self.MatchReplaceBinary.create(1,2,"hallo"), self.MatchReplaceBinary("hallohallohallo"))
        self.assertEqual(self.MatchReplaceBinary.create(-1,"hallo"), self.MatchReplaceBinary("ollah"))
        self.assertEqual(self.MatchReplaceBinary.create(-3,"hallo"), self.MatchReplaceBinary("ollahollahollah"))
        self.assertEqual(self.MatchReplaceBinary.create(1,2,3), self.MatchReplaceBinary(6))


    def testCheckSignature(self):
        self.assertEqual(self.CheckSignature.create(1,"a"), self.CheckSignature(1,"a"))
        self.assertEqual(self.CheckSignature.create(1.,"a"), self.CheckSignature(1.,"a"))
        self.assertRaises(WrongSignatureError, self.CheckSignature.create, "hallo")
        self.assertEqual(self.CheckSignatureFlat.create(1,2.,3j), self.CheckSignatureFlat(1,2.,3j))
        self.assertRaises(WrongSignatureError, self.CheckSignatureFlat.create, 1, 2, "hallo")


    def testIdem(self):
        self.assertEqual(self.Idem.create(2,3,3,1,2,3,4,1,2), self.Idem(1,2,3,4))





