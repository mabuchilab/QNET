.. _printing:

===================
The Printing System
===================

.. currentmodule:: qnet.printing

Overview
--------

As a computer algebra framework, QNET puts great emphasis on the appropriate
display of expressions, both in the context of a Jupyter notebook (QNETs main
"graphical interface") and in the terminal. It also provides the possibility
for you to completely customize the display.

The printing system is modeled closely after the printing system of SymPy (and
directly builds on it). Unlike SymPy, however, the display of an expression
will always directly reflect the algebraic structure (summands will not be
reordered, for example).

In the context of a Jupyter notebook, expressions will be shown via LaTeX.
In an interactive (I)Python terminal, a unicode rendering will be used if the
terminal has unicode support, with a fallback to ascii::

    >>> from qnet.algebra import Create, CoherentStateKet
    >>> from sympy import symbols
    >>> Create(hs='q_1') * CoherentStateKet(symbols('eta')**2/2, hs='q_1')
    â^(q₁)† |α=η²/2⟩^(q₁)

These textual renderings can be obtained manually through the :func:`ascii` and
:func:`unicode` functions.

Unlike SymPy, the unicode rendering will not span multiple lines. Also, QNET
will not rationalize the denominators of scalar fractions by default, to match
the standard notation in quantum mechanics::

    >>> from qnet.algebra import BasisKet
    >>> from sympy import sqrt
    >>> (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)
    1/√2 * (|0⟩⁽¹⁾ + |1⟩⁽¹⁾)

Compare this to the default in SymPy::

    >>> (symbols('a') + symbols('b')) / sqrt(2)
    sqrt(2)*(a + b)/2

With the default settings, the LaTeX renderer that produces the output in the
Jupyter notebook uses only tex macros that MathJax_ understands. You can obtain
the LaTeX code through the :func:`latex` function. When generating code for a
paper or report, it is better to customize the output for better readability
with a more semantic use of macros, e.g. as::

    >>> from qnet.printing import latex
    >>> print(latex((BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2), tex_use_braket=True))
    \frac{1}{\sqrt{2}} \left(\Ket{0}^{(1)} + \Ket{1}^{(1)}\right)

.. _MathJax: https://www.mathjax.org

In addition to the "mathematical" display of expressions, QNET also has
functions to show the exact internal (tree) structure of an expression, either
for debugging or for designing algebraic transformations.

The :func:`srepr` function returns the most direct representation of the
expression: it is a string (possibly with indentation for the tree structure)
that if evaluated results in the exact same expression.

An alternative, specifically for interactive use, is the
:func:`~qnet.printing.tree.print_tree` function. To generate a graphic
representation of the tree structure, the :func:`~qnet.printing.dot.dotprint`
function produces a graph in the DOT language.


Basic Customization
-------------------

At the beginning of an interactive session or notebook, the
:func:`init_printing` routine should be called. This routine associates specific
printing functions, e.g. :func:`unicode`, with the ``__str__`` and ``__repr__``
representation of an expression. It also specifies the default settings for each
printing function. For example, you could suppress the display of Hilbert space
labels::

    >>> from qnet.printing import init_printing
    >>> init_printing(show_hs_label=False)
    >>> (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)
    1/√2 * (|0⟩ + |1⟩)

Or, in a debugging session, you could switch the default representation to use
the indented :func:`srepr`::

    >>> init_printing(repr_format='indsrepr')
    >>> (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)
    ScalarTimesKet(
        Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2))),
        KetPlus(
            BasisKet(
                0,
                hs=LocalSpace(
                    '1')),
            BasisKet(
                1,
                hs=LocalSpace(
                    '1'))))

The settings can also be changed *temporarily* via the :func:`configure_printing`
context manager.


Printer classes
---------------

The printing functions :func:`ascii`, :func:`unicode`, and :func:`latex` each
delegate to an internal printer object that subclasses
:class:`qnet.printing.base.QnetBasePrinter`. For example, the :func:`ascii`
function delegates to an instance of
:class:`qnet.printing._ascii.QnetAsciiPrinter` (after :func:`init_printing` was
called, that printer is located at :attr:`ascii.printer`)

For the ultimate control in customizing the printing system, you can implement your
own subclasses of :class:`~qnet.printing.base.QnetBasePrinter`, which is in
turn a subclass of :class:`sympy.printing.printer.Printer`. Thus, the
overview of `SymPy's printing system`_ applies.

The QNET printers conceptually extend SymPy printers in the following ways:

* QNET printers have support for caching. One reason for this is efficiency.
  The more important reason is that one can use a pre-initialized cache to
  force certain expressions to be represented by fixed strings, which can make
  things considerably more reasonable, and aids in generating code from
  expressions, see the example for :func:`srepr` .
* Every printer contains a sub-printer in the `_sympy_printer` attribute,
  instantiated from the  `sympy_printer_cls` class attribute. Actual SymPy
  objects (e.g., scalar coefficients) are delegated to this sub-printer, while
  the main printer handles all
  :class:`~qnet.algebra.abstract_algebra.Expression` instances. Not that the
  default sub-printers use classes from :mod:`qnet.printing.sympy` that
  implement some custom printing more in line with the conventions of quantum
  physics.

.. _SymPy's printing system: http://docs.sympy.org/latest/modules/printing.html#module-sympy.printing.printer


Customization through an INI file
----------------------------------
