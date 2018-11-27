from os import path, remove
import unittest
from qnet.algebra.core.circuit_algebra import (
    Concatenation,
    SeriesProduct,
    CPermutation,
    CircuitSymbol,
    circuit_identity as cid,
    SeriesInverse,
    Feedback,
    SLH,
)
from qnet.algebra.library.fock_operators import Destroy, Create
from qnet.algebra.core.matrix_algebra import Matrix, identity_matrix
from qnet.visualization.circuit_pyx import draw_circuit, GS
from tempfile import gettempdir
import pyx


class InfrastructureAndCases:
    extension = ""

    def setUp(self):
        self.fname = path.join(gettempdir(), "tmp" + self.extension)

    def tearDown(self):
        if path.exists(self.fname):
            remove(self.fname)

    def testPyX(self):
        if path.exists(self.fname):
            remove(self.fname)

        c = pyx.canvas.canvas()

        c.text(0, 0, "Hello, world!")
        c.stroke(pyx.path.line(0, 0, 2, 0))
        self.write(c)
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

    def assertCanBeDrawn(self, circuit):
        if path.exists(self.fname):
            remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, "lr"))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

        self.assertTrue(draw_circuit(circuit, self.fname, "rl"))
        self.assertTrue(path.exists(self.fname))
        remove(self.fname)

    def testDrawSymbol(self):
        self.assertCanBeDrawn(CircuitSymbol("b", cdim=1))
        self.assertCanBeDrawn(CircuitSymbol("b", cdim=2))

    def testDrawSeriesInverse(self):
        self.assertCanBeDrawn(SeriesInverse(CircuitSymbol("b", cdim=1)))
        self.assertCanBeDrawn(SeriesInverse(CircuitSymbol("b", cdim=2)))

    def testDrawCPermutation(self):
        self.assertCanBeDrawn(CPermutation((0, 2, 3, 1)))

    def testDrawSeries(self):
        self.assertCanBeDrawn(
            SeriesProduct(CircuitSymbol("a", cdim=5), CircuitSymbol("b", cdim=5))
        )

    def testDrawConcatenation(self):
        self.assertCanBeDrawn(
            Concatenation(CircuitSymbol("a", cdim=5), CircuitSymbol("b", cdim=5))
        )

    def testDrawIdentity(self):
        self.assertCanBeDrawn(cid(5))

    def testDrawFeedback(self):
        self.assertCanBeDrawn(
            Feedback(CircuitSymbol("M", cdim=5), out_port=3, in_port=4)
        )

    def testDrawNested(self):
        self.assertCanBeDrawn(
            SeriesProduct(
                CircuitSymbol("a", cdim=2),
                Concatenation(CircuitSymbol("b", cdim=1), CircuitSymbol("c", cdim=1)),
            )
        )
        self.assertCanBeDrawn(
            Concatenation(
                CircuitSymbol("a", cdim=2),
                SeriesProduct(CircuitSymbol("b", cdim=1), CircuitSymbol("c", cdim=1)),
            )
        )
        self.assertCanBeDrawn(
            Feedback(
                Concatenation(
                    CircuitSymbol("a", cdim=2),
                    SeriesProduct(
                        CircuitSymbol("b", cdim=1), CircuitSymbol("c", cdim=1)
                    ),
                ),
                out_port=2,
                in_port=0,
            )
        )

    def testDrawSLH(self):
        self.assertCanBeDrawn(
            SLH(
                identity_matrix(1),
                Matrix([[Create(hs=1)]]),
                Create(hs=1) * Destroy(hs=1),
            )
        )


class TestVisualizationPNG(InfrastructureAndCases, unittest.TestCase):
    extension = ".png"

    def write(self, canvas):
        canvas.writeGSfile(self.fname, gs=GS)


class TestVisualizationEPS(InfrastructureAndCases, unittest.TestCase):
    extension = ".eps"

    def write(self, canvas):
        canvas.writeEPSfile(self.fname)


class TestVisualizationPDF(InfrastructureAndCases, unittest.TestCase):
    extension = ".pdf"

    def write(self, canvas):
        canvas.writePDFfile(self.fname)


if __name__ == "__main__":
    unittest.main()
