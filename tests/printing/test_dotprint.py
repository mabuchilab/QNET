import re
from textwrap import dedent

import pytest

from qnet.algebra.core.scalar_algebra import ScalarValue
from qnet.algebra.core.abstract_algebra import Expression, Operation
from qnet.algebra.core.operator_algebra import OperatorSymbol, Adjoint
from qnet.algebra.core.abstract_quantum_algebra import QuantumSymbol
from qnet.printing import ascii, srepr, configure_printing
from qnet.printing.dot import dotprint, expr_labelfunc


def compare_line(line1, line2, rx=None, match_keys=None):
    """Compare two lines according to the given regex"""
    if rx is None:
        return line1 == line2
    else:
        m1 = rx.match(line1)
        m2 = rx.match(line2)
        if match_keys is None:
            match_keys = m1.groups()
        try:
            for (key, val) in m1.groupdict().items():
                if key in match_keys:
                    if m2.group(key) != val:
                        return False
        except (AttributeError, IndexError, ValueError):
            return False
        return True


def compare_dotcode(dot1, dot2):
    """Compare two multiline dot code strings, ignoring differences in id
    strings"""
    rx_nodeline = re.compile(r'^"(?P<id>[^"]+)" (?P<props>\[.*\]);$')
    rx_edgeline = re.compile(r'^"(?P<id1>[^"]+)" -> "(?P<id2>[^"]+)"')
    res = True
    for (line1, line2) in zip(dot1.splitlines(), dot2.splitlines()):
        lines_match = (
            compare_line(line1, line2, rx_nodeline, ['props']) or
            compare_line(line1, line2, rx_edgeline, []) or
            compare_line(line1, line2))
        if not lines_match:
            print("Lines do not match:\n%s\n%s\n" % (line1, line2))
            res = False
    return res


@pytest.fixture
def expr():
    """Example test expression"""
    A = OperatorSymbol("A", hs=1)
    B = OperatorSymbol("B", hs=1)
    C = OperatorSymbol("C", hs=1)
    D = OperatorSymbol("D", hs=1)

    return 2 * (A + Adjoint(2 * (A + B) + C) + D)


def test_dot_idfunc(expr):
    """Test that using 'str' idfunc and the default idfunc yield matching
    results."""
    dot1 = dotprint(expr)
    dot2 = dotprint(expr, idfunc=str)
    assert compare_dotcode(dot1, dot2)


def test_dot_ascii_id(expr):
    """Test the default dot-rendering of expr (except for using 'str' idfunc,
    since the default 'hash' is non-stable)"""
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, idfunc=ascii)
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "2 * (A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H)_(0, 0)" ["label"="ScalarTimesOperator"];
    "2_(1, 0)" ["label"="2"];
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H_(1, 1)" ["label"="OperatorPlus"];
    "A^(1)_(2, 0)" ["label"="Â⁽¹⁾"];
    "D^(1)_(2, 1)" ["label"="D̂⁽¹⁾"];
    "(C^(1) + 2 * (A^(1) + B^(1)))^H_(2, 2)" ["label"="Adjoint"];
    "C^(1) + 2 * (A^(1) + B^(1))_(3, 0)" ["label"="OperatorPlus"];
    "C^(1)_(4, 0)" ["label"="Ĉ⁽¹⁾"];
    "2 * (A^(1) + B^(1))_(4, 1)" ["label"="ScalarTimesOperator"];
    "2_(5, 0)" ["label"="2"];
    "A^(1) + B^(1)_(5, 1)" ["label"="OperatorPlus"];
    "A^(1)_(6, 0)" ["label"="Â⁽¹⁾"];
    "B^(1)_(6, 1)" ["label"="B̂⁽¹⁾"];

    #########
    # Edges #
    #########

    "2 * (A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H)_(0, 0)" -> "2_(1, 0)"
    "2 * (A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H)_(0, 0)" -> "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H_(1, 1)"
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H_(1, 1)" -> "A^(1)_(2, 0)"
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H_(1, 1)" -> "D^(1)_(2, 1)"
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H_(1, 1)" -> "(C^(1) + 2 * (A^(1) + B^(1)))^H_(2, 2)"
    "(C^(1) + 2 * (A^(1) + B^(1)))^H_(2, 2)" -> "C^(1) + 2 * (A^(1) + B^(1))_(3, 0)"
    "C^(1) + 2 * (A^(1) + B^(1))_(3, 0)" -> "C^(1)_(4, 0)"
    "C^(1) + 2 * (A^(1) + B^(1))_(3, 0)" -> "2 * (A^(1) + B^(1))_(4, 1)"
    "2 * (A^(1) + B^(1))_(4, 1)" -> "2_(5, 0)"
    "2 * (A^(1) + B^(1))_(4, 1)" -> "A^(1) + B^(1)_(5, 1)"
    "A^(1) + B^(1)_(5, 1)" -> "A^(1)_(6, 0)"
    "A^(1) + B^(1)_(5, 1)" -> "B^(1)_(6, 1)"
    }''').strip()


def test_dot_maxdepth2(expr):
    """Test dot-representation with restricted depth"""
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, maxdepth=2)
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "node_(0, 0)" ["label"="ScalarTimesOperator"];
    "node_(1, 0)" ["label"="2"];
    "node_(1, 1)" ["label"="OperatorPlus"];
    "node_(2, 0)" ["label"="Â⁽¹⁾"];
    "node_(2, 1)" ["label"="D̂⁽¹⁾"];
    "node_(2, 2)" ["label"="(Ĉ⁽¹⁾ + 2 (Â⁽¹⁾ + B̂⁽¹⁾))^†"];

    #########
    # Edges #
    #########

    "node_(0, 0)" -> "node_(1, 0)"
    "node_(0, 0)" -> "node_(1, 1)"
    "node_(1, 1)" -> "node_(2, 0)"
    "node_(1, 1)" -> "node_(2, 1)"
    "node_(1, 1)" -> "node_(2, 2)"
    }''').strip()


def test_dot_show_args(expr):
    """Test dot-representation where children are args, not just operands.

    This tests custom 'children', 'is_leaf', and rendering of Expression kwargs
    """

    def _expr_args(expr):
        if isinstance(expr, Expression):
            return expr.args
        else:
            return []

    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, get_children=_expr_args)
    expected = dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "node_(0, 0)" ["label"="ScalarTimesOperator"];
    "node_(1, 0)" ["label"="ScalarValue"];
    "node_(1, 1)" ["label"="OperatorPlus"];
    "node_(2, 0)" ["label"="2"];
    "node_(2, 1)" ["label"="OperatorSymbol(..., hs=ℌ₁)"];
    "node_(2, 2)" ["label"="OperatorSymbol(..., hs=ℌ₁)"];
    "node_(2, 3)" ["label"="Adjoint"];
    "node_(3, 0)" ["label"="A"];
    "node_(3, 1)" ["label"="D"];
    "node_(3, 2)" ["label"="OperatorPlus"];
    "node_(4, 0)" ["label"="OperatorSymbol(..., hs=ℌ₁)"];
    "node_(4, 1)" ["label"="ScalarTimesOperator"];
    "node_(5, 0)" ["label"="C"];
    "node_(5, 1)" ["label"="ScalarValue"];
    "node_(5, 2)" ["label"="OperatorPlus"];
    "node_(6, 0)" ["label"="2"];
    "node_(6, 1)" ["label"="OperatorSymbol(..., hs=ℌ₁)"];
    "node_(6, 2)" ["label"="OperatorSymbol(..., hs=ℌ₁)"];
    "node_(7, 0)" ["label"="A"];
    "node_(7, 1)" ["label"="B"];

    #########
    # Edges #
    #########

    "node_(0, 0)" -> "node_(1, 0)"
    "node_(0, 0)" -> "node_(1, 1)"
    "node_(1, 0)" -> "node_(2, 0)"
    "node_(1, 1)" -> "node_(2, 1)"
    "node_(1, 1)" -> "node_(2, 2)"
    "node_(1, 1)" -> "node_(2, 3)"
    "node_(2, 1)" -> "node_(3, 0)"
    "node_(2, 2)" -> "node_(3, 1)"
    "node_(2, 3)" -> "node_(3, 2)"
    "node_(3, 2)" -> "node_(4, 0)"
    "node_(3, 2)" -> "node_(4, 1)"
    "node_(4, 0)" -> "node_(5, 0)"
    "node_(4, 1)" -> "node_(5, 1)"
    "node_(4, 1)" -> "node_(5, 2)"
    "node_(5, 1)" -> "node_(6, 0)"
    "node_(5, 2)" -> "node_(6, 1)"
    "node_(5, 2)" -> "node_(6, 2)"
    "node_(6, 1)" -> "node_(7, 0)"
    "node_(6, 2)" -> "node_(7, 1)"
    }''').strip()
    assert dotstr.strip() == expected


def test_dot_no_repeat(expr):
    """Test dot-representation with repeating identical nodes"""
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, idfunc=ascii, repeat=False)
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "2 * (A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H)" ["label"="ScalarTimesOperator"];
    "2" ["label"="2"];
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H" ["label"="OperatorPlus"];
    "A^(1)" ["label"="Â⁽¹⁾"];
    "D^(1)" ["label"="D̂⁽¹⁾"];
    "(C^(1) + 2 * (A^(1) + B^(1)))^H" ["label"="Adjoint"];
    "C^(1) + 2 * (A^(1) + B^(1))" ["label"="OperatorPlus"];
    "C^(1)" ["label"="Ĉ⁽¹⁾"];
    "2 * (A^(1) + B^(1))" ["label"="ScalarTimesOperator"];
    "2" ["label"="2"];
    "A^(1) + B^(1)" ["label"="OperatorPlus"];
    "A^(1)" ["label"="Â⁽¹⁾"];
    "B^(1)" ["label"="B̂⁽¹⁾"];

    #########
    # Edges #
    #########

    "2 * (A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H)" -> "2"
    "2 * (A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H)" -> "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H"
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H" -> "A^(1)"
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H" -> "D^(1)"
    "A^(1) + D^(1) + (C^(1) + 2 * (A^(1) + B^(1)))^H" -> "(C^(1) + 2 * (A^(1) + B^(1)))^H"
    "(C^(1) + 2 * (A^(1) + B^(1)))^H" -> "C^(1) + 2 * (A^(1) + B^(1))"
    "C^(1) + 2 * (A^(1) + B^(1))" -> "C^(1)"
    "C^(1) + 2 * (A^(1) + B^(1))" -> "2 * (A^(1) + B^(1))"
    "2 * (A^(1) + B^(1))" -> "2"
    "2 * (A^(1) + B^(1))" -> "A^(1) + B^(1)"
    "A^(1) + B^(1)" -> "A^(1)"
    "A^(1) + B^(1)" -> "B^(1)"
    }''').strip()


def test_dot_custom_labelfunc(expr):
    """Test dot-representation with custom labelfunc"""
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, labelfunc=expr_labelfunc(srepr, str))
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "node_(0, 0)" ["label"="ScalarTimesOperator"];
    "node_(1, 0)" ["label"="ScalarValue(2)"];
    "node_(1, 1)" ["label"="OperatorPlus"];
    "node_(2, 0)" ["label"="OperatorSymbol('A', hs=LocalSpace('1'))"];
    "node_(2, 1)" ["label"="OperatorSymbol('D', hs=LocalSpace('1'))"];
    "node_(2, 2)" ["label"="Adjoint"];
    "node_(3, 0)" ["label"="OperatorPlus"];
    "node_(4, 0)" ["label"="OperatorSymbol('C', hs=LocalSpace('1'))"];
    "node_(4, 1)" ["label"="ScalarTimesOperator"];
    "node_(5, 0)" ["label"="ScalarValue(2)"];
    "node_(5, 1)" ["label"="OperatorPlus"];
    "node_(6, 0)" ["label"="OperatorSymbol('A', hs=LocalSpace('1'))"];
    "node_(6, 1)" ["label"="OperatorSymbol('B', hs=LocalSpace('1'))"];

    #########
    # Edges #
    #########

    "node_(0, 0)" -> "node_(1, 0)"
    "node_(0, 0)" -> "node_(1, 1)"
    "node_(1, 1)" -> "node_(2, 0)"
    "node_(1, 1)" -> "node_(2, 1)"
    "node_(1, 1)" -> "node_(2, 2)"
    "node_(2, 2)" -> "node_(3, 0)"
    "node_(3, 0)" -> "node_(4, 0)"
    "node_(3, 0)" -> "node_(4, 1)"
    "node_(4, 1)" -> "node_(5, 0)"
    "node_(4, 1)" -> "node_(5, 1)"
    "node_(5, 1)" -> "node_(6, 0)"
    "node_(5, 1)" -> "node_(6, 1)"
    }
    ''').strip()


def test_dot_no_styles(expr):
    """Test dot-representation with emtpy 'styles'"""
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, styles=[])
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "node_(0, 0)" ["label"="ScalarTimesOperator"];
    "node_(1, 0)" ["label"="2"];
    "node_(1, 1)" ["label"="OperatorPlus"];
    "node_(2, 0)" ["label"="Â⁽¹⁾"];
    "node_(2, 1)" ["label"="D̂⁽¹⁾"];
    "node_(2, 2)" ["label"="Adjoint"];
    "node_(3, 0)" ["label"="OperatorPlus"];
    "node_(4, 0)" ["label"="Ĉ⁽¹⁾"];
    "node_(4, 1)" ["label"="ScalarTimesOperator"];
    "node_(5, 0)" ["label"="2"];
    "node_(5, 1)" ["label"="OperatorPlus"];
    "node_(6, 0)" ["label"="Â⁽¹⁾"];
    "node_(6, 1)" ["label"="B̂⁽¹⁾"];

    #########
    # Edges #
    #########

    "node_(0, 0)" -> "node_(1, 0)"
    "node_(0, 0)" -> "node_(1, 1)"
    "node_(1, 1)" -> "node_(2, 0)"
    "node_(1, 1)" -> "node_(2, 1)"
    "node_(1, 1)" -> "node_(2, 2)"
    "node_(2, 2)" -> "node_(3, 0)"
    "node_(3, 0)" -> "node_(4, 0)"
    "node_(3, 0)" -> "node_(4, 1)"
    "node_(4, 1)" -> "node_(5, 0)"
    "node_(4, 1)" -> "node_(5, 1)"
    "node_(5, 1)" -> "node_(6, 0)"
    "node_(5, 1)" -> "node_(6, 1)"
    }
    ''').strip()


def test_dot_custom_styles(expr):
    """Test dot-representation with custom styles"""
    styles = [
        (lambda expr: isinstance(expr, QuantumSymbol),
            {'color': 'red', 'shape': 'box', 'fontsize': 12}),
        (lambda expr: isinstance(expr, ScalarValue),
            {'color': 'blue', 'shape': 'box', 'fontsize': 12}),
        (lambda expr: isinstance(expr, Operation),
            {'color': 'black', 'shape': 'ellipse'})]
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(expr, styles=styles)
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "ordering"="out"
    "rankdir"="TD"

    #########
    # Nodes #
    #########

    "node_(0, 0)" ["color"="black", "label"="ScalarTimesOperator", "shape"="ellipse"];
    "node_(1, 0)" ["color"="blue", "fontsize"="12", "label"="2", "shape"="box"];
    "node_(1, 1)" ["color"="black", "label"="OperatorPlus", "shape"="ellipse"];
    "node_(2, 0)" ["color"="red", "fontsize"="12", "label"="Â⁽¹⁾", "shape"="box"];
    "node_(2, 1)" ["color"="red", "fontsize"="12", "label"="D̂⁽¹⁾", "shape"="box"];
    "node_(2, 2)" ["color"="black", "label"="Adjoint", "shape"="ellipse"];
    "node_(3, 0)" ["color"="black", "label"="OperatorPlus", "shape"="ellipse"];
    "node_(4, 0)" ["color"="red", "fontsize"="12", "label"="Ĉ⁽¹⁾", "shape"="box"];
    "node_(4, 1)" ["color"="black", "label"="ScalarTimesOperator", "shape"="ellipse"];
    "node_(5, 0)" ["color"="blue", "fontsize"="12", "label"="2", "shape"="box"];
    "node_(5, 1)" ["color"="black", "label"="OperatorPlus", "shape"="ellipse"];
    "node_(6, 0)" ["color"="red", "fontsize"="12", "label"="Â⁽¹⁾", "shape"="box"];
    "node_(6, 1)" ["color"="red", "fontsize"="12", "label"="B̂⁽¹⁾", "shape"="box"];

    #########
    # Edges #
    #########

    "node_(0, 0)" -> "node_(1, 0)"
    "node_(0, 0)" -> "node_(1, 1)"
    "node_(1, 1)" -> "node_(2, 0)"
    "node_(1, 1)" -> "node_(2, 1)"
    "node_(1, 1)" -> "node_(2, 2)"
    "node_(2, 2)" -> "node_(3, 0)"
    "node_(3, 0)" -> "node_(4, 0)"
    "node_(3, 0)" -> "node_(4, 1)"
    "node_(4, 1)" -> "node_(5, 0)"
    "node_(4, 1)" -> "node_(5, 1)"
    "node_(5, 1)" -> "node_(6, 0)"
    "node_(5, 1)" -> "node_(6, 1)"
    }
    ''').strip()


def test_dot_custom_kwargs(expr):
    """Test dot-representation with custom kwargs for graph attributes"""
    with configure_printing(str_format='unicode', repr_format='unicode'):
        dotstr = dotprint(
            expr, rankdir='LR', splines='curved',
            label='Expression Tree')
    assert dotstr.strip() == dedent(r'''
    digraph{

    # Graph style
    "label"="Expression Tree"
    "ordering"="out"
    "rankdir"="LR"
    "splines"="curved"

    #########
    # Nodes #
    #########

    "node_(0, 0)" ["label"="ScalarTimesOperator"];
    "node_(1, 0)" ["label"="2"];
    "node_(1, 1)" ["label"="OperatorPlus"];
    "node_(2, 0)" ["label"="Â⁽¹⁾"];
    "node_(2, 1)" ["label"="D̂⁽¹⁾"];
    "node_(2, 2)" ["label"="Adjoint"];
    "node_(3, 0)" ["label"="OperatorPlus"];
    "node_(4, 0)" ["label"="Ĉ⁽¹⁾"];
    "node_(4, 1)" ["label"="ScalarTimesOperator"];
    "node_(5, 0)" ["label"="2"];
    "node_(5, 1)" ["label"="OperatorPlus"];
    "node_(6, 0)" ["label"="Â⁽¹⁾"];
    "node_(6, 1)" ["label"="B̂⁽¹⁾"];

    #########
    # Edges #
    #########

    "node_(0, 0)" -> "node_(1, 0)"
    "node_(0, 0)" -> "node_(1, 1)"
    "node_(1, 1)" -> "node_(2, 0)"
    "node_(1, 1)" -> "node_(2, 1)"
    "node_(1, 1)" -> "node_(2, 2)"
    "node_(2, 2)" -> "node_(3, 0)"
    "node_(3, 0)" -> "node_(4, 0)"
    "node_(3, 0)" -> "node_(4, 1)"
    "node_(4, 1)" -> "node_(5, 0)"
    "node_(4, 1)" -> "node_(5, 1)"
    "node_(5, 1)" -> "node_(6, 0)"
    "node_(5, 1)" -> "node_(6, 1)"
    }
    ''').strip()
