# encoding: utf-8
"""
test_algebra.py

Created by Nikolas Tezak on 2011-02-08.
Copyright (c) 2011 . All rights reserved.
"""

import sys
import os
import unittest
from algebra.operator_algebra import *
import algebra.abstract_algebra


# def symbols(*names):
#     return [ScalarSymbol(name) for name in names]
#     
# a,b,c = symbols("a", "b", "c")
# 
# 
# algebra.abstract_algebra.CHECK_OPERANDS = True
# algebra.abstract_algebra.PRINT_PRETTY = True
# 
# 
# class TestAddition(unittest.TestCase):
#     def testRules(self):
#         self.assertEqual(a+0, a)
#         self.assertEqual(0+a, a)
#         self.assertEqual(1 + a + 1, a + 2)
#         self.assertEqual(2 + a, ScalarAddition(a, 2))
#         self.assertEqual(a + 2 + b, ScalarAddition(a, b, 2))
#         self.assertRaises(Exception, ScalarAddition, (a,))
#         self.assertRaises(Exception, ScalarAddition, ())
#         self.assertEqual(ScalarAddition.apply_with_rules(a,-a),0)
#         self.assertEqual(a + b +a, 2*a + b)
#         self.assertEqual(a/b + 5 + a/c, 5 + (a*b + a*c)/(b*c))
# 
# class TestMultiplication(unittest.TestCase):
#     def testRules(self):
#         self.assertRaises(Exception, ScalarMultiplication, ())
#         self.assertRaises(Exception, ScalarMultiplication, (a))
#         self.assertEqual(a*b, b*a)
#         self.assertEqual(a*b, ScalarMultiplication(a, b))
#         self.assertEqual(a*(a/b), (a*a)/b)
#         self.assertEqual((a/c)*(a/b), (a*a)/(b*c))
# 
# 
# class TestNumberScalarProduct(unittest.TestCase):
#     def testRules(self):
#         self.assertEqual(a*1,a)
#         self.assertEqual(1*a, a)        
#         self.assertEqual(a*5,NumberScalarProduct(5, a))
#         self.assertEqual(5*a,NumberScalarProduct(5, a))
#         self.assertEqual(2*a*3, 6*a)        
#         self.assertEqual(a*5*b, NumberScalarProduct(5, a*b))
#         self.assertEqual(NumberScalarProduct.apply_with_rules(1.,a), a)
#         self.assertEqual(NumberScalarProduct.apply_with_rules(1,a), a)
#         self.assertEqual(NumberScalarProduct.apply_with_rules(1+0j, a), a)
#         self.assertEqual(0 * a, 0)
#         self.assertEqual(a*0, 0)
# 
# class TestFractions(unittest.TestCase):
#     def testRules(self):
#         self.assertEqual(a/b, ScalarFraction(a,b))
#         self.assertEqual(a*(b/c), (a*b)/c)
#         
#     def testCancellation(self):
#         f = a/b
#         finv = b/a
#         g = b/c
#         self.assertEqual(f, ScalarFraction(a,b))
#         self.assertEqual(f*finv, 1)
#         self.assertEqual(f*g, a/c)
#         self.assertEqual((5*a) / (10*b) , ScalarFraction(a, 2*b))
#         self.assertEqual((a*a)/a, a)
#         
#         
# class TestPowers(unittest.TestCase):
#     def testRules(self):
#         self.assertEqual(a*a, a**2)
#         self.assertEqual(a**b, ScalarPower(a,b))
#         self.assertEqual((a**b)**c, ScalarPower(a,b*c))
#         self.assertEqual(a**(b**c), ScalarPower(a, ScalarPower(b, c)))
#         self.assertEqual(a**1, a)
#         self.assertEqual(1**a, 1)
#         self.assertEqual(a**0, 1)
#         self.assertEqual(a**2 * b * a, a ** 3 * b)
#     
#     def testPowerProducts(self):
#         self.assertEqual((a**b)*(a**c), a**(b+c))
# 
# 
# class TestExp(unittest.TestCase):
#     
#     
#     def testRules(self):
#         self.assertEqual(exp(a), ScalarExp(a))
#         self.assertEqual(exp(0), 1)
#         self.assertEqual(exp(1+1j).evalf(), math.exp(1) * (math.cos(1) + 1j * math.sin(1)))
#         
#     def testExpPower(self):
#         a,b,c = symbols("a","b","c")        
#         self.assertEqual(exp(a)**2, exp(2*a))
#         self.assertEqual(exp(a)**b, exp(a*b))
#     
#     def testExpProducts(self):
#         self.assertEqual(exp(a)*exp(b), exp(a+b))
#     
#     def testExpQuotients(self):
#         self.assertEqual(1/exp(a), exp(-a))
#         self.assertEqual(exp(a)*(1/exp(b)), exp(a-b))
#         self.assertEqual(exp(a)/exp(b), exp(a-b))        
#         self.assertEqual(exp(a)*exp(b)/exp(c), exp(a+b-c))
# 
# class TestSqrt(unittest.TestCase):
#     def testRules(self):
#         self.assertEqual(sqrt(0),0)
#         self.assertEqual(sqrt(1800**2), 1800)
#         self.assertRaises(OperandsError, sqrt(a).substitute, {a:-10})
#         self.assertRaises(OperandsError, sqrt, -20)
# 
# 
# 
#         
# 
# 
# 
# if __name__ == '__main__':
#   unittest.main()
# 
