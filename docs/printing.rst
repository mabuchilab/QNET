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
terminal has unicode support, with a fallback to ascii. We can force this
manually by::

    >>> init_printing(repr_format='unicode')

    >>> Create(hs='q_1') * CoherentStateKet(symbols('eta')**2/2, hs='q_1')
    â^(q₁)† |α=η²/2⟩^(q₁)

These textual renderings can be obtained manually through the :func:`ascii` and
:func:`unicode` functions.

Unlike SymPy, the unicode rendering will not span multiple lines. Also, QNET
will not rationalize the denominators of scalar fractions by default, to match
the standard notation in quantum mechanics::

    >>> (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)
    1/√2 (|0⟩⁽¹⁾ + |1⟩⁽¹⁾)

Compare this to the default in SymPy::

    >>> (symbols('a') + symbols('b')) / sqrt(2)  # doctest: +SKIP
    √2⋅(a + b)
    ──────────
        2

With the default settings, the LaTeX renderer that produces the output in the
Jupyter notebook uses only tex macros that MathJax_ understands. You can obtain
the LaTeX code through the :func:`latex` function. When generating code for a
paper or report, it is better to customize the output for better readability
with a more semantic use of macros, e.g. as::

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
:func:`~qnet.printing.treeprinting.print_tree` function. To generate a graphic
representation of the tree structure, the :func:`~qnet.printing.dot.dotprint`
function produces a graph in the DOT language.


Basic Customization
-------------------

At the beginning of an interactive session or notebook, the
:func:`init_printing` routine should be called. This routine associates specific
printing functions, e.g. :func:`unicode`, with the ``__str__`` and ``__repr__``
representation of an expression. This is what is returned by ``str(expr)``, and
by ``repr(expr)`` or as the output in an interactive (I)Python session.
The initialization also specifies the default settings for each
printing function. For example, you could suppress the display of Hilbert space
labels::

    >>> init_printing(show_hs_label=False, repr_format='unicode')
    >>> (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)
    1/√2 (|0⟩ + |1⟩)

Or, in a debugging session, you could switch the default representation to use
the indented :func:`srepr`::

    >>> init_printing(repr_format='indsrepr')              # doctest: +SKIP
    >>> (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)  # doctest: +SKIP
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

Note that :func:`init_printing` should only be called once; or else it should
be given the ``reset`` parameter::

    >>> init_printing(repr_format='unicode', reset=True)


Printer classes
---------------

The printing functions :func:`ascii`, :func:`unicode`, and :func:`latex` each
delegate to an internal printer object that subclasses
:class:`qnet.printing.base.QnetBasePrinter`. After initialization,
the printer class is referenced at e.g. :attr:`ascii.printer`.

For the ultimate control in customizing the printing system, you can implement your
own subclasses of :class:`~qnet.printing.base.QnetBasePrinter`, which is in
turn a subclass of :class:`sympy.printing.printer.Printer`. Thus, the
overview of `SymPy's printing system`_ applies.

The QNET printers conceptually extend SymPy printers in the following ways:

* QNET printers have support for caching. One reason for this is efficiency.
  More importantly, it allows to pass a pre-initialized cache to
  force certain expressions to be represented by fixed strings, which can make
  expressions considerably more readable, and aids in generating code from
  expressions, see the example for :func:`srepr` .
* Every printer contains a sub-printer in the `_sympy_printer` attribute,
  instantiated from the  `sympy_printer_cls` class attribute. Actual SymPy
  objects (e.g., scalar coefficients) are delegated to this sub-printer, while
  the main printer handles all
  :class:`~qnet.algebra.abstract_algebra.Expression` instances. Not that the
  default sub-printers use classes from :mod:`qnet.printing.sympy` that
  implement some custom printing more in line with the conventions of quantum
  physics.

When :func:`init_printing` is called with direct settings as in the previous
section, these will be used as
*global* settings, and will affect any printers (including SymPy sub-printers)
that are instantiated afterwards.

The settings that are given to any printing function will be used for that
specific call of the printing function only. If you define custom classes with
different or additional settings and set them up for use with the printing
function (see below), the accepted arguments to the printing functions change
accordingly.

.. _SymPy's printing system: http://docs.sympy.org/latest/modules/printing.html#module-sympy.printing.printer


.. _ini_file_printing:

Customization through an INI file
----------------------------------

While :func:`init_printing` can simply be called with explicit settings to configure the
printing system globally (see above), for a more advanced set up an INI-file
can be used. In this case, the path to the file must be the only argument::

    init_printing(inifile=<path to file>)

This allows to associate custom printer classes with the printing functions,
and also define the settings settings for those particular printers (as opposed
to just global settings).

The INI file may have sections 'global', 'ascii', 'unicode', and 'latex'.
Parameters in the 'global' section are equivalent to those could be passed to
:func:`init_printing` as direct settings. That is, they set up the printing
function to be used for ``__str__`` and ``__repr__``, and set the global
options for all printer classes.

The 'ascii', 'unicode', and 'latex' sections configure the respective printing
functions. To link them to custom Printer classes, you may specify ``printer``
and ``sympy_printer`` as the full path to the Printer class that should be used
for the main printer and the sub-printer for SymPy expressions. All other
settings in the sections override the settings from 'global' for that
particular printer.

Consider the following annotated example for an INI file::

    [global]
    # The settings in the 'global' section are for all Printer classes (both
    # SymPy and QNET). They are equivalent to passing them to init_printing
    # directly

    # the printing function to use for str(expr)
    str_format = ascii
    # the printing function to use for expr(expr)
    repr_format = unicode
    # direct global settings
    show_hs_label = False
    sig_as_ketbra = False
    # note that boolean values must be specified as "True", or "False"

    # The three sections below associate the printing functions with particular
    # Printer classes, and override the global settings for those particular
    # printers

    [ascii]
    printer = qnet.printing.asciiprinter.QnetAsciiPrinter
    # we use the SymPy StrPrinter here, instead of the default
    # qnet.printing.sympy.SympyStrPrinter that is customized to not
    # rationalize denominators
    sympy_printer = sympy.printing.str.StrPrinter
    # we override the the settings from the 'global' section
    show_hs_label = True
    sig_as_ketbra = True

    [unicode]
    printer = qnet.printing.unicodeprinter.QnetUnicodePrinter
    sympy_printer = qnet.printing.sympy.SympyUnicodePrinter
    show_hs_label = subscript
    unicode_op_hats = False

    [latex]
    printer = qnet.printing.latexprinter.QnetLatexPrinter
    sympy_printer = qnet.printing.sympy.SympyLatexPrinter
    # string values can be written un-escaped
    tex_op_macro = \Op{{{name}}}
    tex_use_braket = True
    # You can also include options for the sympy_printer
    inv_trig_style = full
