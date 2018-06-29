from os.path import join

import pytest

from sympy import symbols, sqrt, atan
from sympy.printing.str import StrPrinter
from sympy.printing.printer import Printer

from qnet.utils.testing import datadir, QnetAsciiTestPrinter

from qnet.algebra.core.operator_algebra import LocalSigma
from qnet.algebra.core.state_algebra import BasisKet
from qnet.printing import (
    init_printing, configure_printing, ascii, unicode, latex)
from qnet.printing.asciiprinter import QnetAsciiPrinter
from qnet.printing.unicodeprinter import QnetUnicodePrinter
from qnet.printing.sympy import SympyStrPrinter


datadir = pytest.fixture(datadir)


def test_initfile(datadir):
    psi = (BasisKet(0, hs=1) + BasisKet(1, hs=1)) / sqrt(2)
    x = symbols('x')
    sig = LocalSigma(0, 1, hs=1)

    init_printing(str_format='unicode', repr_format='unicode')

    str(psi) == unicode(psi)
    repr(psi) == unicode(psi)
    assert isinstance(ascii.printer, QnetAsciiPrinter)
    assert isinstance(ascii.printer._sympy_printer, SympyStrPrinter)
    assert isinstance(unicode.printer, QnetUnicodePrinter)
    assert ascii(psi) == '1/sqrt(2) * (|0>^(1) + |1>^(1))'
    assert unicode(psi) == '1/√2 (|0⟩⁽¹⁾ + |1⟩⁽¹⁾)'
    assert (
        latex(psi) ==
        r'\frac{1}{\sqrt{2}} \left(\left\lvert 0 \right\rangle^{(1)} + '
        r'\left\lvert 1 \right\rangle^{(1)}\right)')
    assert (
        latex(atan(x) * sig) ==
        r'\operatorname{atan}{\left (x \right )} \left\lvert 0 '
        r'\middle\rangle\!\middle\langle 1 \right\rvert^{(1)}')

    with configure_printing(inifile=join(datadir, 'printing.ini')):
        assert (
            Printer._global_settings['val1'] ==
            '1 # inline comments are not allowd')
        assert (
            Printer._global_settings['val2'] ==
            '1 ; with either prefix character')
        assert 'show_hs_label' in Printer._global_settings
        assert 'sig_as_ketbra' in Printer._global_settings
        assert 'use_unicode' in Printer._global_settings
        assert len(Printer._global_settings) == 5
        str(psi) == ascii(psi)
        repr(psi) == unicode(psi)
        assert isinstance(ascii.printer, QnetAsciiTestPrinter)
        assert isinstance(ascii.printer._sympy_printer, StrPrinter)
        assert isinstance(unicode.printer, QnetUnicodePrinter)
        assert ascii(psi) == 'sqrt(2)/2 * (|0>^(1) + |1>^(1))'
        assert unicode(psi) == '1/√2 (|0⟩₍₁₎ + |1⟩₍₁₎)'
        assert (
            latex(psi) ==
            r'\frac{1}{\sqrt{2}} \left(\Ket{0} + \Ket{1}\right)')
        assert (
            latex(atan(x) * sig) ==
            r'\arctan{\left (x \right )} \Op{\sigma}_{0,1}')

    assert 'use_unicode' in Printer._global_settings
    assert len(Printer._global_settings) == 1

    str(psi) == unicode(psi)
    repr(psi) == unicode(psi)
    assert isinstance(ascii.printer, QnetAsciiPrinter)
    assert isinstance(ascii.printer._sympy_printer, SympyStrPrinter)
    assert isinstance(unicode.printer, QnetUnicodePrinter)
    assert ascii(psi) == '1/sqrt(2) * (|0>^(1) + |1>^(1))'
    assert unicode(psi) == '1/√2 (|0⟩⁽¹⁾ + |1⟩⁽¹⁾)'
    assert (
        latex(psi) ==
        r'\frac{1}{\sqrt{2}} \left(\left\lvert 0 \right\rangle^{(1)} + '
        r'\left\lvert 1 \right\rangle^{(1)}\right)')
    assert (
        latex(atan(x) * sig) ==
        r'\operatorname{atan}{\left (x \right )} \left\lvert 0 '
        r'\middle\rangle\!\middle\langle 1 \right\rvert^{(1)}')

    init_printing(reset=True)


def test_inifile_do_not_mix(datadir):
    with pytest.raises(TypeError) as exc_info:
        init_printing(
            str_format='ascii', repr_format='ascii',
            inifile=join(datadir, 'printing.ini'))
        assert (
            "The `inifile` argument cannot be combined with any other "
            "keyword arguments" in str(exc_info.value))


def test_invalid_section(datadir):
    with pytest.raises(ValueError) as exc_info:
        init_printing(inifile=join(datadir, 'invalid_section.ini'))
    assert "Invalid section sympy" in str(exc_info.value)
    init_printing(reset=True)


def test_invalid_options(datadir):
    with pytest.raises(TypeError) as exc_info:
        init_printing(inifile=join(datadir, 'invalid_value.ini'))
    assert (
        "some_bogus_setting is not a valid setting for either "
        "QnetAsciiTestPrinter or StrPrinter" in str(exc_info.value))
    init_printing(reset=True)
