#!/usr/bin/env python
# encoding: utf-8
"""
test_algebra.py

Created by Nikolas Tezak on 2011-02-08.
Copyright (c) 2011 . All rights reserved.
"""

import unittest
from qnet.algebra.circuit_algebra import *
import qnet.algebra.abstract_algebra


symbol_counter = 0
def get_symbol(cdim):
    global symbol_counter
    sym =  CSymbol('test_%d' % symbol_counter, cdim)
    symbol_counter +=1
    return sym
def get_symbols(*cdim):
    return [get_symbol(n) for n in cdim]
    

qnet.algebra.abstract_algebra.CHECK_OPERANDS = True
qnet.algebra.abstract_algebra.PRINT_PRETTY = True


class TestPermutations(unittest.TestCase):
    def testPermutation(self):
        n = 5
        
        self.assertEqual(CPermutation(()), circuit_identity(0))
        invalid_permutation = (1,1)
        self.assertRaises(Exception, CPermutation, (invalid_permutation,))
        p_id = range(n)
        self.assertEqual(CPermutation(p_id), circuit_identity(n))
    
        
        self.assertEqual(map_signals({0:1,1:0}, 2), (1,0))
        self.assertEqual(map_signals({0:5,1:0}, 6), (5,0,1,2,3,4))
        
        self.assertEqual(map_signals({0:5,1:0, 3:2}, 6), invert_permutation(map_signals({5:0,0:1, 2:3}, 6)))
        
        
        

class TestCircuitAlgebra(unittest.TestCase):
    
    def testSeries(self):
        A, B = get_symbol(1), get_symbol(1)
        self.assertEquals( A << B, SeriesProduct(A,B))
        self.assertEquals( A<< B, SeriesProduct.create(A,B))
        
        # need at least two operands
        self.assertRaises(Exception, SeriesProduct, ())
        self.assertRaises(Exception, SeriesProduct.create, ())
        self.assertRaises(Exception, SeriesProduct, (A,))
        
        self.assertEquals(SeriesProduct.create(A), A)

    
    def testSeriesFilterIdentities(self):
        for n in (1,2,3, 10):
            A, B = get_symbol(n), get_symbol(n)
            idn = circuit_identity(n)
            self.assertEquals(A << idn, A)
            self.assertEquals(idn << A, A)
            self.assertEquals(SeriesProduct.create(idn, idn, A, idn, idn, B, idn, idn), A << B)
    
    def testConcatenation(self):
        n = 4
        A, B = get_symbol(n), get_symbol(n)
        id0 = circuit_identity(0)
        self.assertEquals(A+B, Concatenation(A,B))
        self.assertEquals(A+B, Concatenation.create(A,B))
        self.assertEquals(id0 + id0 + A + id0 + id0 + B + id0 + id0, A + B)
        self.assertRaises(Exception, Concatenation, ())
        self.assertRaises(Exception, Concatenation, (A,))
        
        self.assertEquals((A+B).block_structure, (n,n))
        
        self.assertEquals((A+B).get_blocks((n,n)), (A,B))
        #test index_in_block()
        self.assertEquals((A+B).index_in_block(0), (0,0))
        self.assertEquals((A+B).index_in_block(1), (1,0))
        self.assertEquals((A+B).index_in_block(2), (2,0))
        self.assertEquals((A+B).index_in_block(3), (3,0))
        self.assertEquals((A+B).index_in_block(4), (0,1))
        self.assertEquals((A+B).index_in_block(5), (1,1))
        self.assertEquals((A+B).index_in_block(7), (3,1))

    def testDistributiveLaw(self):
         A, B, C, D, E = get_symbols(2,1,1,1,1)
         
         self.assertEquals((A+B) << (C+D+E), Concatenation(A<<(C+D), B << E))
         
         self.assertEquals((C+D+E) << (A+B) , Concatenation((C+D)<< A,  E<< B))
         
         self.assertEquals((A+B) << (C+D+E) << (A+B) , Concatenation(A << (C+D)<< A,  B << E<< B))
    
         self.assertEquals(SeriesProduct.create((A+B), (C+D+E), (A+B)), Concatenation(A << (C+D)<< A,  B << E<< B))
         
         test_perm = (0,1,3,2)
         qtp = CPermutation(test_perm)
         
         self.assertEquals(CPermutation((1,0)) << ( B + C), SeriesProduct(Concatenation(C, B), CPermutation((1,0))))
         
         self.assertEquals(qtp << (A + B + C), (A + C+ B) <<  qtp)
         
         self.assertEquals(qtp << ( B + C + A) , B + C + (CPermutation((1,0)) << A))
         
         test_perm2 = (1,0,3,2)
         qtp2 = CPermutation(test_perm2)
         
         self.assertEquals(qtp2 << (A + B + C), (CPermutation((1,0)) << A) + ((C+B) << CPermutation((1,0))))
         
         self.assertEquals(qtp << qtp2, CPermutation(CPermutation.permute(test_perm, test_perm2)))
        
    def testCPermutation(self):
        test_perm = (0,1,2,5,6,3,4)
        qtp = CPermutation(test_perm)
        self.assertEqual(qtp, CPermutation(list(test_perm)))
        self.assertEqual(qtp.series_inverse(), CPermutation(invert_permutation(test_perm)))
        self.assertEqual(qtp.block_structure, (1,1,1,4))
        id1 = circuit_identity(1)
        self.assertEqual(qtp.get_blocks(), (id1, id1, id1, CPermutation((2,3,0,1))))
            
        self.assertEqual(CPermutation((1,0,3,2)).get_blocks(), (CPermutation((1,0)), CPermutation((1,0))))
        nt = len(test_perm)
        self.assertEqual(qtp << qtp.series_inverse(), circuit_identity(nt))
        self.assertEqual(CPermutation.permute(list(invert_permutation(test_perm)), test_perm), range(nt))  
        
    def testFactorizePermutation(self):
        self.assertEqual(CPermutation.full_block_perm((0,1,2), (1,1,1)), (0,1,2))
        self.assertEqual(CPermutation.full_block_perm((0,2,1), (1,1,1)), (0,2,1))
        self.assertEqual(CPermutation.full_block_perm((0,2,1), (1,1,2)), (0,3,1,2))
        self.assertEqual(CPermutation.full_block_perm((0,2,1), (1,2,3)), (0,4,5,1,2,3))
        self.assertEqual(CPermutation.full_block_perm((1,2,0), (1,2,3)), (3,4,5,0,1,2))
        self.assertEqual(CPermutation.full_block_perm((3,1,2,0), (1,2,3,4)), (9, 4, 5, 6, 7, 8, 0, 1, 2, 3 ))
        self.assertEqual(CPermutation.block_perm_and_perms_within_blocks((9, 4, 5, 6, 7, 8, 0, 1, 2, 3 ), (1,2,3,4)), \
                                                                        ((3,1,2,0), [(0,),(0,1),(0,1,2),(0,1,2,3)]))
    
        A1,A2,A3,A4 = get_symbols(1,2,3,4)

        new_lhs, permuted_rhs, new_rhs = P_sigma(9, 4, 5, 6, 7, 8, 0, 1, 2, 3 ).factorize_for_rhs(A1+A2+A3+A4)
        self.assertEqual(new_lhs, cid(10))
        self.assertEqual(permuted_rhs, (A4+A2+A3+A1))
        self.assertEqual(new_rhs, P_sigma(9, 4, 5, 6, 7, 8, 0, 1, 2, 3 ))

        p = P_sigma(0,1,4,2,3,5)
        expr = A2 + A3 + A1
        new_lhs, permuted_rhs, new_rhs = p.factorize_for_rhs(expr)
        self.assertEqual(new_lhs, cid(6))
        self.assertEqual(permuted_rhs, A2 + (P_sigma(2,0,1) << A3) + A1)
        self.assertEqual(new_rhs, cid(6))
        
        
        p = P_sigma(0, 3, 1, 2)
        
        p_r = P_sigma(2, 0, 1)
        assert p == cid(1) + p_r
        A = get_symbol(2)
        
        

        new_lhs, permuted_rhs, new_rhs = p.factorize_for_rhs(cid(1) + A+ cid(1))

        self.assertEqual(new_lhs, P_sigma(0,1,3,2))
        self.assertEqual(permuted_rhs, (cid(1) + (P_sigma(1,0) << A)  + cid(1)))
        self.assertEqual(new_rhs, cid(4))
        
        
        new_lhs, permuted_rhs, new_rhs = p.factorize_for_rhs(cid(2) + A)
        
        self.assertEqual(new_lhs, cid(4))
        self.assertEqual(permuted_rhs, (cid(1) + A  + cid(1)))
        self.assertEqual(new_rhs, p)
        
        self.assertEqual(p.series_inverse() << (cid(2) + A), cid(1) + SeriesProduct(P_sigma(0,2,1), Concatenation(SeriesProduct(P_sigma(1,0), A), cid(1)),P_sigma(2,0,1)))
        
        self.assertEqual(p.series_inverse() << (cid(2) + A) << p, cid(1) + (p_r.series_inverse() << (cid(1) + A) << p_r))

        new_lhs, permuted_rhs, new_rhs = P_sigma(4,2,1,3,0).factorize_for_rhs((A4 + cid(1)))
        self.assertEqual(new_lhs, cid(5))
        self.assertEqual(permuted_rhs, (cid(1) + (P_sigma(3,1,0,2) << A4)))
        self.assertEqual(new_rhs, map_signals_circuit({4:0}, 5))
        
        
        ## special test case that helped find the major permutation block structure factorization bug
        p = P_sigma(3, 4, 5, 0, 1, 6, 2) 
        q = cid(3) + CSymbol('NAND1', 4)
        
        new_lhs, permuted_rhs, new_rhs = p.factorize_for_rhs(q)
        self.assertEqual(new_lhs, P_sigma(0,1,2,6,3,4,5))
        self.assertEqual(permuted_rhs, (P_sigma(0,1,3,2) << CSymbol('NAND1', 4)) + cid(3))
        self.assertEqual(new_rhs, P_sigma(4,5,6, 0,1,2,3))

        
    
    
    def testFeedback(self):
        A, B, C, D, A1, A2 = get_symbols(3,2,1,1,1,1)
        
        n = 4
        cid1 = circuit_identity(1)
        
        self.assertRaises(Exception, Feedback, ())
        self.assertRaises(Exception, Feedback, (C,))
        self.assertRaises(Exception, Feedback, (C + D,))
        self.assertRaises(Exception, Feedback, (C << D,))
        self.assertRaises(Exception, Feedback, (circuit_identity(n),))
        self.assertRaises(Exception, Feedback.create, (circuit_identity(0)))
        self.assertEquals(Feedback.create(circuit_identity(n)), circuit_identity(n-1))
        self.assertEquals(Feedback.create(A+B), A + Feedback.create(B))
        smq = map_signals_circuit({2:1}, 3) # == 'cid(1) + X'
        self.assertEquals(smq, smq.series_inverse())
        # import metapost as mp
        # mp.display_circuit(Feedback.apply_with_rules(smq.series_inverse() << (B + C) << smq))
        # mp.display_circuit(B.feedback() + C)
        
        self.assertEquals(( smq << (B + C)).feedback(out_index = 2, in_index = 1), B.feedback() + C)        
        self.assertEquals(( smq << (B + C) << smq).feedback(), B.feedback() + C)

        self.assertEquals((B + C).feedback(1,1), B.feedback() + C)
        
        #check that feedback is resolved into series when possible
        self.assertEquals(B.feedback(1,0).substitute({B:(C+D)}), C << D)
        self.assertEquals((A << (B + cid(1))).feedback(),  A.feedback() << B)
        self.assertEquals((A << (B + cid(1)) << (cid(1) + P_sigma(1,0))).feedback(2,1),  A.feedback() << B)
        self.assertEquals((A << (cid(1) + P_sigma(1,0)) << (B + cid(1)) << (cid(1) + P_sigma(1,0))).feedback(1,1),  A.feedback(1,1) << B)
        self.assertEquals((B << (cid(1)  + C)).feedback(0,1).substitute({B: (A1 + A2)}), A2 << C << A1)
        self.assertEquals(((cid(1)  + C)<< P_sigma(1,0) << B).feedback(1,1).substitute({B: (A1 + A2)}), A2 << C << A1)
        self.assertEquals(((cid(1)  + C)<< P_sigma(1,0) << B << (cid(1) + D)).feedback(1,1).substitute({B: (A1 + A2)}), A2 << D<< C << A1)
        
        
        
        # self.assertEquals(Feedback.apply_with_rules)
        
        # self.assertEquals()
        
    def testSpecialCases(self):
        pass
        # testobjs = Concatenation(CIdentity(),CPermutation((0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2)))
        
        # self.assertEquals(testobjs, Concatenation.apply_with_rules(*testobjs.operands))
        
    # def testTripletts(self):
        
        

def main():
    unittest.main()

if __name__ == '__main__':
    main()
    # suite = unittest.TestLoader().loadTestsFromTestCase()

