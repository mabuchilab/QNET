"""Printing system for QNET Expressions and related objects"""

import sys
import logging
import configparser
import importlib
from contextlib import contextmanager
from collections import defaultdict
from functools import partial

from sympy.interactive.printing import init_printing as sympy_init_printing
from sympy.printing.printer import Printer as SympyPrinter

from .base import QnetBasePrinter
from .asciiprinter import QnetAsciiPrinter
from .unicodeprinter import QnetUnicodePrinter
from .latexprinter import QnetLatexPrinter
from .sreprprinter import QnetSReprPrinter, IndentedSReprPrinter
from .treeprinting import print_tree, tree
from .dot import dotprint

__all__ = ['init_printing', 'configure_printing', 'ascii', 'unicode', 'latex',
           'tex', 'srepr', 'dotprint', 'tree', 'print_tree']
__private__ = []


def _printer_cls(label, class_address, require_base=QnetBasePrinter):
    module_name, class_name = class_address.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError):
        raise ValueError("%s '%s' does not exist" % (label, class_address))
    try:
        if require_base is not None:
            if not issubclass(cls, require_base):
                raise ValueError(
                    "%s '%s' must be a subclass of %s"
                    % (label, class_address, require_base.__name__))
    except TypeError:
            raise ValueError(
                "%s '%s' must be a class" % (label, class_address))
    else:
        return cls


def init_printing(*, reset=False, init_sympy=True, **kwargs):
    """Initialize the printing system.

    This determines the behavior of the :func:`ascii`, :func:`unicode`,
    and :func:`latex` functions, as well as the ``__str__`` and ``__repr__`` of
    any :class:`.Expression`.

    The routine may be called in one of two forms. First,

    ::

        init_printing(
            str_format=<str_fmt>, repr_format=<repr_fmt>,
            caching=<use_caching>, **settings)

    provides a simplified, "manual" setup with the following parameters.

    Args:
        str_format (str): Format for ``__str__`` representation of an
            :class:`.Expression`. One of 'ascii', 'unicode', 'latex', 'srepr',
            'indsrepr' ("indented `srepr`"), or 'tree'. The string
            representation will be affected by the settings for the
            corresponding print routine, e.g. :func:`unicode` for
            ``str_format='unicode'``
        repr_format (str): Like `str_format`, but for ``__repr__``. This is
            what gets displayed in an interactive (I)Python session.
        caching (bool): By default, the printing functions  (:func:`ascii`,
            :func:`unicode`, :func:`latex`) cache their result for any
            expression and sub-expression. This is both for efficiency and to
            give the ability to to supply custom strings for subexpression by
            passing a `cache` parameter to the printing functions. Initializing
            the printing system with ``caching=False`` disables this
            possibility.
        settings: Any setting understood by any of the printing routines.

    Second,

    ::

        init_printing(inifile=<path_to_file>)


    allows for more detailed settings through a config file, see the
    :ref:`notes on using an INI file <ini_file_printing>`.

    If `str_format` or `repr_format` are not given, they will be set to
    'unicode' if the current terminal is known to support an UTF8 (accordig to
    ``sys.stdout.encoding``), and 'ascii' otherwise.

    Generally, :func:`init_printing` should be called only once at the
    beginning of a script or notebook. If it is called multiple times, any
    settings accumulate. To avoid this and to reset the printing system to the
    defaults, you may pass ``reset=True``.  In a Jupyter notebook, expressions
    are rendered graphically via LaTeX, using the settings as they affect the
    :func:`latex` printer.

    The :func:`sympy.init_printing()` routine is called automatically, unless
    `init_sympy` is given as ``False``.

    See also:
        :func:`configure_printing` allows to temporarily change the printing
        system from what was configured in :func:`init_printing`.
    """
    # return either None (default) or a dict of frozen attributes if
    # ``_freeze=True`` is given as a keyword argument (internal use in
    # `configure_printing` only)
    logger = logging.getLogger(__name__)
    if reset:
        SympyPrinter._global_settings = {}
    if init_sympy:
        if kwargs.get('repr_format', '') == 'unicode':
            sympy_init_printing(use_unicode=True)
        if kwargs.get('repr_format', '') == 'ascii':
            sympy_init_printing(use_unicode=False)
        else:
            sympy_init_printing()  # let sympy decide by itself
    if 'inifile' in kwargs:
        invalid_kwargs = False
        if '_freeze' in kwargs:
            _freeze = kwargs['_freeze']
            if len(kwargs) != 2:
                invalid_kwargs = True
        else:
            _freeze = False
            if len(kwargs) != 1:
                invalid_kwargs = True
        if invalid_kwargs:
            raise TypeError(
                "The `inifile` argument cannot be combined with any "
                "other keyword arguments")
        logger.debug(
            "Initializating printing from INI file %s", kwargs['inifile'])
        return _init_printing_from_file(kwargs['inifile'], _freeze=_freeze)
    else:
        logger.debug(
            "Initializating printing with direct settings: %s", repr(kwargs))
        return _init_printing(**kwargs)


def _init_printing(
        str_format=None, repr_format=None, caching=True,
        ascii_printer='qnet.printing.asciiprinter.QnetAsciiPrinter',
        ascii_sympy_printer='qnet.printing.sympy.SympyStrPrinter',
        ascii_settings=None,
        unicode_printer='qnet.printing.unicodeprinter.QnetUnicodePrinter',
        unicode_sympy_printer='qnet.printing.sympy.SympyUnicodePrinter',
        unicode_settings=None,
        latex_printer='qnet.printing.latexprinter.QnetLatexPrinter',
        latex_sympy_printer='qnet.printing.sympy.SympyLatexPrinter',
        latex_settings=None, _freeze=False, **settings):
    # Note: the *_printer args are undocumented, it's preferable to use them
    # through an INI file only
    logger = logging.getLogger(__name__)
    freeze = defaultdict(dict)
    freeze[SympyPrinter]['_global_settings'] \
        = SympyPrinter._global_settings.copy()
    # Putting the settings in the _global_settings dict for SympyPrinter makes
    # sure that any setting that is acceptable to any Printer that is newly
    # instantiated is used automatically. Settings in _global_settings that are
    # not in the _default_settings of a Printer will be silently ignored.
    SympyPrinter.set_global_settings(**settings)
    # Note that this is the *only* mechanism by which we handle the settings;
    # no settings are passed to any specific Printer below -- Printer-specific
    # settings are possible only when using an INI file.

    freeze[QnetBasePrinter]['_allow_caching'] = QnetBasePrinter._allow_caching
    QnetBasePrinter._allow_caching = caching

    print_map = {
        # print fct     printer cls     sympy printer class           settings
        'ascii':   (ascii_printer,   ascii_sympy_printer,   ascii_settings),
        'unicode': (unicode_printer, unicode_sympy_printer, unicode_settings),
        'latex':   (latex_printer,   latex_sympy_printer,   latex_settings),
    }

    for name in print_map.keys():

        print_func = _PRINT_FUNC[name]
        qnet_printer_address, sympy_printer_address, settings = print_map[name]
        if settings is None:
            settings = {}

        if hasattr(print_func, '_printer_cls'):
            freeze[print_func]['_printer_cls'] = print_func._printer_cls
            freeze[print_func._printer_cls]['sympy_printer_cls'] = \
                print_func._printer_cls.sympy_printer_cls
        if hasattr(print_func, 'printer'):
            freeze[print_func]['printer'] = print_func.printer

        print_func._printer_cls = _printer_cls(
            name + '_printer', qnet_printer_address)
        print_func._printer_cls.sympy_printer_cls = _printer_cls(
            name + '_sympy_printer', sympy_printer_address,
            require_base=SympyPrinter)
        # instantiation of sympy_printer happens in init-routine (which is why
        # the sympy_printer_cls must be set first!)
        print_func.printer = print_func._printer_cls(settings=settings)

    # set up the __str__ and __repr__ printers
    try:
        has_unicode = "UTF-8" in sys.stdout.encoding
    except TypeError:
        has_unicode = False
    logger.debug(
        "Terminal supports unicode: %s (autodetect)", has_unicode)
    if repr_format == 'unicode':
        has_unicode = True
    _PRINT_FUNC['tree'] = partial(tree, unicode=has_unicode)
    if str_format is None:
        str_format = 'unicode' if has_unicode else 'ascii'
        logger.debug("Setting __str__ format to %s", str_format)
    try:
        str_func = _PRINT_FUNC[str_format]
    except KeyError:
        raise ValueError(
            "str_format must be one of %s" % ", ".join(_PRINT_FUNC.keys()))
    if repr_format is None:
        repr_format = 'unicode' if has_unicode else 'ascii'
        logger.debug("Setting __repr__ format to %s" % repr_format)
    try:
        repr_func = _PRINT_FUNC[repr_format]
    except KeyError:
        raise ValueError(
            "repr_format must be one of %s" % ", ".join(_PRINT_FUNC.keys()))
    from qnet.algebra.core.abstract_algebra import Expression
    freeze[Expression]['__str__'] = Expression.__str__
    freeze[Expression]['__repr__'] = Expression.__repr__
    freeze[Expression]['_repr_latex_'] = Expression._repr_latex_
    Expression.__str__ = lambda self: str_func(self)
    Expression.__repr__ = lambda self: repr_func(self)
    Expression._repr_latex_ = lambda self: "$" + latex(self) + "$"
    if _freeze:
        return freeze


def _init_printing_from_file(inifile, _freeze=False):
    config = configparser.ConfigParser(interpolation=None)
    config.BOOLEAN_STATES = {
        'true': True, 'True': True, 'false': False, 'False': False}
    config.read(inifile)
    kwargs = defaultdict(dict)
    for section in config.sections():
        if section == 'DEFAULT':
            continue
        allowed_sections = ['global', 'ascii', 'unicode', 'latex']
        if section not in allowed_sections:
            raise ValueError(
                "Invalid section %s in %s. Allowed sections are %s"
                % (section, inifile, ", ".join(allowed_sections)))
        for key, val in config[section].items():
            if val in config.BOOLEAN_STATES:
                val = config.BOOLEAN_STATES[val]
            if section == 'global':
                kwargs[key] = val
            else:
                if key == 'printer':
                    kwargs["%s_printer" % section] = val
                elif key == 'sympy_printer':
                    kwargs["%s_sympy_printer" % section] = val
                else:
                    kwargs["%s_settings" % section][key] = val
    return _init_printing(_freeze=_freeze, **dict(kwargs))


@contextmanager
def configure_printing(**kwargs):
    """Context manager for temporarily changing the printing system.

    This takes the same parameters as :func:`init_printing`

    Example:

        >>> A = OperatorSymbol('A', hs=1); B = OperatorSymbol('B', hs=1)
        >>> with configure_printing(show_hs_label=False):
        ...     print(ascii(A + B))
        A + B
        >>> print(ascii(A + B))
        A^(1) + B^(1)
    """
    freeze = init_printing(_freeze=True, **kwargs)
    try:
        yield
    finally:
        for obj, attr_map in freeze.items():
            for attr, val in attr_map.items():
                setattr(obj, attr, val)


def ascii(expr, cache=None, **settings):
    """Return an ASCII representation of the given object / expression

    Args:
        expr: Expression to print
        cache (dict or None): dictionary to use for caching
        show_hs_label (bool or str): Whether to a label for the Hilbert space
            of `expr`. By default (``show_hs_label=True``), the label is shown
            as a superscript. It can be shown as a subscript with
            ``show_hs_label='subscript'`` or suppressed entirely
            (``show_hs_label=False``)
        sig_as_ketbra (bool): Whether to render instances of
            :class:`.LocalSigma` as a ket-bra (default), or as an operator
            symbol

    Examples:

        >>> A = OperatorSymbol('A', hs=1); B = OperatorSymbol('B', hs=1)
        >>> ascii(A + B)
        'A^(1) + B^(1)'
        >>> ascii(A + B, cache={A: 'A', B: 'B'})
        'A + B'
        >>> ascii(A + B, show_hs_label='subscript')
        'A_(1) + B_(1)'
        >>> ascii(A + B, show_hs_label=False)
        'A + B'
        >>> ascii(LocalSigma(0, 1, hs=1))
        '|0><1|^(1)'
        >>> ascii(LocalSigma(0, 1, hs=1), sig_as_ketbra=False)
        'sigma_0,1^(1)'

    Note that the accepted parameters and their default values may be changed
    through :func:`init_printing` or :func:`configure_printing`
    """
    try:
        if cache is None and len(settings) == 0:
            return ascii.printer.doprint(expr)
        else:
            printer = ascii._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        ascii._printer_cls = QnetAsciiPrinter
        ascii.printer = ascii._printer_cls()
        return ascii(expr, cache, **settings)


def unicode(expr, cache=None, **settings):
    """Return a unicode representation of the given object / expression

    Args:
        expr: Expression to print
        cache (dict or None): dictionary to use for caching
        show_hs_label (bool or str): Whether to a label for the Hilbert space
            of `expr`. By default (``show_hs_label=True``), the label is shown
            as a superscript. It can be shown as a subscript with
            ``show_hs_label='subscript'`` or suppressed entirely
            (``show_hs_label=False``)
        sig_as_ketbra (bool): Whether to render instances of
            :class:`.LocalSigma` as a ket-bra (default), or as an operator
            symbol
        unicode_sub_super (bool): Whether to try to use unicode symbols for
            sub- or superscripts if possible
        unicode_op_hats (bool): Whether to draw unicode hats on single-letter
            operator symbols

    Examples:

        >>> A = OperatorSymbol('A', hs=1); B = OperatorSymbol('B', hs=1)
        >>> unicode(A + B)
        'Â⁽¹⁾ + B̂⁽¹⁾'
        >>> unicode(A + B, cache={A: 'A', B: 'B'})
        'A + B'
        >>> unicode(A + B, show_hs_label='subscript')
        'Â₍₁₎ + B̂₍₁₎'
        >>> unicode(A + B, show_hs_label=False)
        'Â + B̂'
        >>> unicode(LocalSigma(0, 1, hs=1))
        '|0⟩⟨1|⁽¹⁾'
        >>> unicode(LocalSigma(0, 1, hs=1), sig_as_ketbra=False)
        'σ̂_0,1^(1)'
        >>> unicode(A + B, unicode_sub_super=False)
        'Â^(1) + B̂^(1)'
        >>> unicode(A + B, unicode_op_hats=False)
        'A⁽¹⁾ + B⁽¹⁾'

    Note that the accepted parameters and their default values may be changed
    through :func:`init_printing` or :func:`configure_printing`
    """
    try:
        if cache is None and len(settings) == 0:
            return unicode.printer.doprint(expr)
        else:
            printer = unicode._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        unicode._printer_cls = QnetUnicodePrinter
        unicode.printer = unicode._printer_cls()
        return unicode(expr, cache, **settings)


def latex(expr, cache=None, **settings):
    r"""Return a LaTeX representation of the given object / expression

    Args:
        expr: Expression to print
        cache (dict or None): dictionary to use for caching
        show_hs_label (bool or str): Whether to a label for the Hilbert space
            of `expr`. By default (``show_hs_label=True``), the label is shown
            as a superscript. It can be shown as a subscript with
            ``show_hs_label='subscript'`` or suppressed entirely
            (``show_hs_label=False``)
        tex_op_macro (str): macro to use for formatting operator symbols.
            Must accept 'name' as a format key.
        tex_textop_macro (str): macro to use for formatting multi-letter
            operator names.
        tex_sop_macro (str): macro to use for formattign super-operator symbols
        tex_textsop_macro (str): macro to use for formatting multi-letter
            super-operator names
        tex_identity_sym (str): macro for the identity symbol
        tex_use_braket (bool): If True, use macros from the
            `braket package
            <https://ctan.org/tex-archive/macros/latex/contrib/braket>`_. Note
            that this will not automatically render in IPython Notebooks, but
            it is recommended when generating latex for a document.
        tex_frac_for_spin_labels (bool): Whether to use '\frac' when printing
            basis state labels for spin Hilbert spaces


    Examples:

        >>> A = OperatorSymbol('A', hs=1); B = OperatorSymbol('B', hs=1)
        >>> latex(A + B)
        '\\hat{A}^{(1)} + \\hat{B}^{(1)}'
        >>> latex(A + B, cache={A: 'A', B: 'B'})
        'A + B'
        >>> latex(A + B, show_hs_label='subscript')
        '\\hat{A}_{(1)} + \\hat{B}_{(1)}'
        >>> latex(A + B, show_hs_label=False)
        '\\hat{A} + \\hat{B}'
        >>> latex(LocalSigma(0, 1, hs=1))
        '\\left\\lvert 0 \\middle\\rangle\\!\\middle\\langle 1 \\right\\rvert^{(1)}'
        >>> latex(LocalSigma(0, 1, hs=1), sig_as_ketbra=False)
        '\\hat{\\sigma}_{0,1}^{(1)}'
        >>> latex(A + B, tex_op_macro=r'\Op{{{name}}}')
        '\\Op{A}^{(1)} + \\Op{B}^{(1)}'
        >>> CNOT = OperatorSymbol('CNOT', hs=1)
        >>> latex(CNOT)
        '\\text{CNOT}^{(1)}'
        >>> latex(CNOT, tex_textop_macro=r'\Op{{{name}}}')
        '\\Op{CNOT}^{(1)}'

        >>> A = SuperOperatorSymbol('A', hs=1)
        >>> latex(A)
        '\\mathrm{A}^{(1)}'
        >>> latex(A, tex_sop_macro=r'\SOp{{{name}}}')
        '\\SOp{A}^{(1)}'
        >>> Lindbladian = SuperOperatorSymbol('Lindbladian', hs=1)
        >>> latex(Lindbladian)
        '\\mathrm{Lindbladian}^{(1)}'
        >>> latex(Lindbladian, tex_textsop_macro=r'\SOp{{{name}}}')
        '\\SOp{Lindbladian}^{(1)}'

        >>> latex(IdentityOperator)
        '\\mathbb{1}'
        >>> latex(IdentityOperator, tex_identity_sym=r'\identity')
        '\\identity'
        >>> latex(LocalSigma(0, 1, hs=1), tex_use_braket=True)
        '\\Ket{0}\\!\\Bra{1}^{(1)}'

        >>> spin = SpinSpace('s', spin=(1, 2))
        >>> up = SpinBasisKet(1, 2, hs=spin)
        >>> latex(up)
        '\\left\\lvert +1/2 \\right\\rangle^{(s)}'
        >>> latex(up, tex_frac_for_spin_labels=True)
        '\\left\\lvert +\\frac{1}{2} \\right\\rangle^{(s)}'

    Note that the accepted parameters and their default values may be changed
    through :func:`init_printing` or :func:`configure_printing`
    """
    try:
        if cache is None and len(settings) == 0:
            return latex.printer.doprint(expr)
        else:
            printer = latex._printer_cls(cache, settings)
            return printer.doprint(expr)
    except AttributeError:
        # init_printing was not called. Setting up defaults
        latex._printer_cls = QnetLatexPrinter
        latex.printer = latex._printer_cls()
        return latex(expr, cache, **settings)


def tex(expr, cache=None, **settings):
    """Alias for :func:`latex`"""
    return latex(expr, cache, **settings)


def srepr(expr, indented=False, cache=None):
    """Render the given expression into a string that can be evaluated in an
    appropriate context to re-instantiate an identical expression. If
    `indented` is False (default), the resulting string is a single line.
    Otherwise, the result is a multiline string, and each positional and
    keyword argument of each `Expression` is on a separate line, recursively
    indented to produce a tree-like output. The `cache` may be used to generate
    more readable expressions.

    Example:

        >>> hs = LocalSpace('1')
        >>> A = OperatorSymbol('A', hs=hs); B = OperatorSymbol('B', hs=hs)
        >>> expr = A + B
        >>> srepr(expr)
        "OperatorPlus(OperatorSymbol('A', hs=LocalSpace('1')), OperatorSymbol('B', hs=LocalSpace('1')))"
        >>> eval(srepr(expr)) == expr
        True
        >>> srepr(expr, cache={hs:'hs'})
        "OperatorPlus(OperatorSymbol('A', hs=hs), OperatorSymbol('B', hs=hs))"
        >>> eval(srepr(expr, cache={hs:'hs'})) == expr
        True
        >>> print(srepr(expr, indented=True))
        OperatorPlus(
            OperatorSymbol(
                'A',
                hs=LocalSpace(
                    '1')),
            OperatorSymbol(
                'B',
                hs=LocalSpace(
                    '1')))
        >>> eval(srepr(expr, indented=True)) == expr
        True

    See also:
        :func:`~qnet.printing.tree.print_tree`, respectively
        :func:`qnet.printing.tree.tree`, produces an output similar to
        the indented :func:`srepr`, for interactive use. Their result
        cannot be evaluated and the exact output depends on
        :func:`init_printing`.

        :func:`~qnet.printing.dot.dotprint` provides a way to graphically
        explore the tree structure of an expression.
    """
    if indented:
        printer = IndentedSReprPrinter(cache=cache)
    else:
        printer = QnetSReprPrinter(cache=cache)
    return printer.doprint(expr)


# Map acceptable values for `str_format` and `repr`_format in
# `init_printing` to a print function
_PRINT_FUNC = {
    'ascii': ascii,
    'unicode': unicode,
    'latex': latex,
    'tex': latex,
    'srepr': srepr,
    'indsrepr': partial(srepr, indented=True),
    'tree': tree,   # init_printing will modify this for unicode support
}
