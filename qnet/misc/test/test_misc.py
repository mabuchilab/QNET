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


from qnet.misc.parse_circuit_strings import parse_circuit_strings
from qnet.algebra.circuit_algebra import Concatenation, SeriesProduct, CPermutation, CircuitSymbol, CIdentity, cid, Feedback, identity_matrix, Create, Destroy, Matrix, SLH
from qnet.misc.circuit_visualization import draw_circuit
from os import path, remove
from tempfile import gettempdir
import pyx

import unittest

class TestCircuitParsing(unittest.TestCase):
    def testSymbol(self):
        self.assertEqual(parse_circuit_strings('a(2)'), CircuitSymbol('a', 2))
        self.assertEqual(parse_circuit_strings('a(5)'), CircuitSymbol('a', 5))
        self.assertEqual(parse_circuit_strings('a_longer_string(5)'), CircuitSymbol('a_longer_string', 5))

    def testCPermutation(self):
        self.assertEqual(parse_circuit_strings('P_sigma(0,2,1)'), CPermutation((0,2,1)))
        self.assertEqual(parse_circuit_strings('P_sigma(0,2,1,4,3)'), CPermutation((0,2,1,4,3)))

    def testSeries(self):
        self.assertEqual(parse_circuit_strings('a(2) <<  b(2)'), SeriesProduct(CircuitSymbol('a',2),CircuitSymbol('b',2)))
        self.assertEqual(parse_circuit_strings('a(5) <<  b(5) << c(5)'), SeriesProduct(CircuitSymbol('a',5),CircuitSymbol('b',5),CircuitSymbol('c',5)))

    def testConcatenation(self):
        self.assertEqual(parse_circuit_strings('a(1) +  b(2)'), Concatenation(CircuitSymbol('a',1),CircuitSymbol('b',2)))
        self.assertEqual(parse_circuit_strings('a(1) +  b(2) + c(3)'), Concatenation(CircuitSymbol('a',1),CircuitSymbol('b',2),CircuitSymbol('c',3)))

    def testCIdentity(self):
        self.assertEqual(parse_circuit_strings('cid(1)'), CIdentity)
        self.assertEqual(parse_circuit_strings('cid(5)'), cid(5))

    def testFeedback(self):
        self.assertEqual(parse_circuit_strings('[M(5)]_(3->4)'), Feedback(CircuitSymbol('M',5), 3, 4))

    def testNested(self):
        self.assertEqual(parse_circuit_strings('a(2) <<  (b(1) + c(1))'), SeriesProduct(CircuitSymbol('a',2),Concatenation(CircuitSymbol('b',1), CircuitSymbol('c',1))))
        self.assertEqual(parse_circuit_strings('a(2) +  (b(1) << c(1))'), Concatenation(CircuitSymbol('a',2),SeriesProduct(CircuitSymbol('b',1), CircuitSymbol('c',1))))
        self.assertEqual(parse_circuit_strings('[a(2) +  (b(1) << c(1))]_(2->0)'), Feedback(Concatenation(CircuitSymbol('a',2),SeriesProduct(CircuitSymbol('b',1), CircuitSymbol('c',1))),2,0))

class TestVisualizationPNG(unittest.TestCase):

    def setUp(self):
        self.fname = gettempdir()  + '/tmp.png'

    def tearDown(self):
        if path.exists(self.fname):
            remove(self.fname)

    def testPyX(self):

        if path.exists(self.fname):
            remove(self.fname)

        c = pyx.canvas.canvas()

        c.text(0, 0, "Hello, world!")
        c.stroke(pyx.path.line(0, 0, 2, 0))
        c.writeGSfile(self.fname)
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)


    def assertCanBeDrawn(self, circuit):

        if path.exists(self.fname):
            remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, 'lr'))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, 'rl'))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

    def testDrawSymbol(self):
        self.assertCanBeDrawn(CircuitSymbol('b',1))
        self.assertCanBeDrawn(CircuitSymbol('b',2))

    def testDrawCPermutation(self):
        self.assertCanBeDrawn(CPermutation((0,2,3,1)))

    def testDrawSeries(self):
        self.assertCanBeDrawn(SeriesProduct(CircuitSymbol('a',5),CircuitSymbol('b',5)))

    def testDrawConcatenation(self):
        self.assertCanBeDrawn(Concatenation(CircuitSymbol('a', 5), CircuitSymbol('b', 5)))

    def testDrawIdentity(self):
        self.assertCanBeDrawn(cid(5))

    def testDrawFeedback(self):
        self.assertCanBeDrawn(Feedback(CircuitSymbol('M',5), 3, 4))

    def testDrawNested(self):
        self.assertCanBeDrawn(SeriesProduct(CircuitSymbol('a',2),Concatenation(CircuitSymbol('b',1), CircuitSymbol('c',1))))
        self.assertCanBeDrawn(Concatenation(CircuitSymbol('a',2),SeriesProduct(CircuitSymbol('b',1), CircuitSymbol('c',1))))
        self.assertCanBeDrawn(Feedback(Concatenation(CircuitSymbol('a',2),SeriesProduct(CircuitSymbol('b',1), CircuitSymbol('c',1))),2,0))

    def testDrawSLH(self):
        self.assertCanBeDrawn(SLH(identity_matrix(1), Matrix([[Create(1)]]), Create(1)*Destroy(1)))

    def testDrawComponent(self):
        from qnet.circuit_components import kerr_cavity_cc as kerr
        K = kerr.KerrCavity()
        self.assertCanBeDrawn(K)
        self.assertCanBeDrawn(K.creduce())
        self.assertCanBeDrawn(K.toSLH())


class TestVisualizationEPS(unittest.TestCase):
    def setUp(self):
        self.fname = gettempdir() + '/tmp.eps'

    def tearDown(self):
        if path.exists(self.fname):
            remove(self.fname)

    def testPyX(self):
        if path.exists(self.fname):
            remove(self.fname)

        c = pyx.canvas.canvas()

        c.text(0, 0, "Hello, world!")
        c.stroke(pyx.path.line(0, 0, 2, 0))
        c.writeEPSfile(self.fname)
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)


    def assertCanBeDrawn(self, circuit):
        if path.exists(self.fname):
            remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, 'lr'))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, 'rl'))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

    def testDrawSymbol(self):
        self.assertCanBeDrawn(CircuitSymbol('b', 1))
        self.assertCanBeDrawn(CircuitSymbol('b', 2))

    def testDrawCPermutation(self):
        self.assertCanBeDrawn(CPermutation((0, 2, 3, 1)))

    def testDrawSeries(self):
        self.assertCanBeDrawn(SeriesProduct(CircuitSymbol('a', 5), CircuitSymbol('b', 5)))

    def testDrawConcatenation(self):
        self.assertCanBeDrawn(Concatenation(CircuitSymbol('a', 5), CircuitSymbol('b', 5)))

    def testDrawIdentity(self):
        self.assertCanBeDrawn(cid(5))

    def testDrawFeedback(self):
        self.assertCanBeDrawn(Feedback(CircuitSymbol('M', 5), 3, 4))

    def testDrawNested(self):
        self.assertCanBeDrawn(
            SeriesProduct(CircuitSymbol('a', 2), Concatenation(CircuitSymbol('b', 1), CircuitSymbol('c', 1))))
        self.assertCanBeDrawn(
            Concatenation(CircuitSymbol('a', 2), SeriesProduct(CircuitSymbol('b', 1), CircuitSymbol('c', 1))))
        self.assertCanBeDrawn(
            Feedback(Concatenation(CircuitSymbol('a', 2), SeriesProduct(CircuitSymbol('b', 1), CircuitSymbol('c', 1))),
                2, 0))

    def testDrawSLH(self):
        self.assertCanBeDrawn(SLH(identity_matrix(1), Matrix([[Create(1)]]), Create(1) * Destroy(1)))

    def testDrawComponent(self):
        from qnet.circuit_components import kerr_cavity_cc as kerr

        K = kerr.KerrCavity()
        self.assertCanBeDrawn(K)
        self.assertCanBeDrawn(K.creduce())
        self.assertCanBeDrawn(K.toSLH())


class TestVisualizationPDF(unittest.TestCase):
    def setUp(self):
        self.fname = gettempdir() + '/tmp.pdf'

    def tearDown(self):
        if path.exists(self.fname):
            remove(self.fname)

    def testPyX(self):
        if path.exists(self.fname):
            remove(self.fname)

        c = pyx.canvas.canvas()

        c.text(0, 0, "Hello, world!")
        c.stroke(pyx.path.line(0, 0, 2, 0))
        c.writePDFfile(self.fname)
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)


    def assertCanBeDrawn(self, circuit):
        if path.exists(self.fname):
            remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, 'lr'))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, 'rl'))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

    def testDrawSymbol(self):
        self.assertCanBeDrawn(CircuitSymbol('b', 1))
        self.assertCanBeDrawn(CircuitSymbol('b', 2))

    def testDrawCPermutation(self):
        self.assertCanBeDrawn(CPermutation((0, 2, 3, 1)))

    def testDrawSeries(self):
        self.assertCanBeDrawn(SeriesProduct(CircuitSymbol('a', 5), CircuitSymbol('b', 5)))

    def testDrawConcatenation(self):
        self.assertCanBeDrawn(Concatenation(CircuitSymbol('a', 5), CircuitSymbol('b', 5)))

    def testDrawIdentity(self):
        self.assertCanBeDrawn(cid(5))

    def testDrawFeedback(self):
        self.assertCanBeDrawn(Feedback(CircuitSymbol('M', 5), 3, 4))

    def testDrawNested(self):
        self.assertCanBeDrawn(
            SeriesProduct(CircuitSymbol('a', 2), Concatenation(CircuitSymbol('b', 1), CircuitSymbol('c', 1))))
        self.assertCanBeDrawn(
            Concatenation(CircuitSymbol('a', 2), SeriesProduct(CircuitSymbol('b', 1), CircuitSymbol('c', 1))))
        self.assertCanBeDrawn(
            Feedback(Concatenation(CircuitSymbol('a', 2), SeriesProduct(CircuitSymbol('b', 1), CircuitSymbol('c', 1))),
                2, 0))

    def testDrawSLH(self):
        self.assertCanBeDrawn(SLH(identity_matrix(1), Matrix([[Create(1)]]), Create(1) * Destroy(1)))

    def testDrawComponent(self):
        from qnet.circuit_components import kerr_cavity_cc as kerr

        K = kerr.KerrCavity()
        self.assertCanBeDrawn(K)
        self.assertCanBeDrawn(K.creduce())
        self.assertCanBeDrawn(K.toSLH())


if __name__ == '__main__':
    unittest.main()