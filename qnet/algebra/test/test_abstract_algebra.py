from qnet.algebra.abstract_algebra import *
import unittest




class Dummy1(Operation):
    pass


class Dummy2(Operation):
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


    def testFlatWithWildcard(self):

        a = wc("a", head = int)
        b = wc("b")

        self.assertEqual(match(a, Dummy1(1,3,2)), Match({a: Dummy1(1,3,2)}))
        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,3,2)), Match({a: 3}))
        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo",2)), Match({b: "hallo"}))
        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1,"hallo","hallo")), Match({b: "hallo"}))


    def testNestedWithWildcards(self):
        a = wc("a", head = int)
        b = wc("b")
        c = wc("c", head = Dummy2)
        self.assertEqual(match(Dummy1(Dummy2(a,1),b,2), Dummy1(Dummy2(1,1),3,2)), Match({a: 1, b: 3}))
        self.assertEqual(match(Dummy1(Dummy2(a,1),a,2), Dummy1(Dummy2(1,1),1,2)), Match({a: 1}))
        self.assertEqual(match(Dummy1(Dummy2(a,1),a,2), Dummy1(Dummy2(1,1),-1,2)), False)
        self.assertEqual(match(Dummy1(c,2), Dummy1(Dummy2(1,2,3))), Match({c: Dummy2(1,2,3)}))


    def testOneOrMore(self):
        a = wc("a__", head = int)
        b = wc("b__")
        self.assertEqual(match(b, Dummy1(1,3,4,2)), Match({b: Dummy1(1,3,4,2)}))
        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,3,4,2)), Match({a: (3,4)}))
        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,3,2)), Match({a: 3}))

        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo",2)), Match({b: "hallo"}))
        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo",1,2)), Match({b: ("hallo",1)}))
        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo","du",2)), Match({b: ("hallo","du")}))

        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,"hallo","du",2)), False)
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1,"hallo","hallo")), Match({b: "hallo"}))

    def testZeroOrMore(self):
        a = wc("a___", head = int)
        b = wc("b___")

        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,2)), Match({a: ()}))
        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,3,2)), Match({a: 3}))

        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,2)), Match({b: ()}))
        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo",2)), Match({b: "hallo"}))
        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo",1,2)), Match({b: ("hallo",1)}))
        self.assertEqual(match(Dummy1(1,b,2), Dummy1(1,"hallo","du",2)), Match({b: ("hallo","du")}))

        self.assertEqual(match(Dummy1(1,a,2), Dummy1(1,"hallo","du",2)), False)
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1,"hallo",2)), False)
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1)), Match({b:()}))
        self.assertEqual(match(Dummy1(1,b,b), Dummy1(1,"hallo","hallo")), Match({b: "hallo"}))


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

        @flat
        class Flat(Operation):
            pass


        @orderless
        class Orderless(Operation):
            pass


        @filter_neutral
        class FilterNeutral(Operation):
            neutral_element = object()


        @check_signature
        class CheckSignature(Operation):
            signature = (int, long, float, complex), str

        @flat
        @check_signature_flat
        class CheckSignatureFlat(Operation):
            signature = (int, long, float, complex),


        def mult_str_int_if_pos(s, i):
            if i >= 0:
                return s * i
            raise CannotSimplify()


        @flat
        @match_replace_binary
        class MatchReplaceBinary(Operation):
            binary_rules = [
                ((wc("a",head = int), wc("b",head = int)), lambda a,b: a + b),
                ((wc("a",head = int), wc("b",head = str)), lambda b,a: mult_str_int_if_pos(b,a))
            ]

        self.Flat = Flat
        self.Orderless = Orderless
        self.FilterNeutral = FilterNeutral
        self.CheckSignature = CheckSignature
        self.CheckSignatureFlat = CheckSignatureFlat
        self.MatchReplaceBinary = MatchReplaceBinary




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
        self.assertEqual(self.MatchReplaceBinary.create(-1,"hallo"), self.MatchReplaceBinary(-1,"hallo"))
        self.assertEqual(self.MatchReplaceBinary.create(1,2,3), self.MatchReplaceBinary(6))


    def testCheckSignature(self):
        self.assertEqual(self.CheckSignature.create(1,"a"), self.CheckSignature(1,"a"))
        self.assertEqual(self.CheckSignature.create(1.,"a"), self.CheckSignature(1.,"a"))
        self.assertRaises(WrongSignature, self.CheckSignature.create, "hallo")
        self.assertEqual(self.CheckSignatureFlat.create(1,2.,3j), self.CheckSignatureFlat(1,2.,3j))
        self.assertRaises(WrongSignature, self.CheckSignatureFlat.create, 1, 2, "hallo")







