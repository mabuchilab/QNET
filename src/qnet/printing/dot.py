"""`DOT`_ printer for Expressions.

This module provides the :func:`dotprint` function that generates a `DOT`_
diagram for a given expression. For example::

    >>> A = OperatorSymbol("A", hs=1)
    >>> B = OperatorSymbol("B", hs=1)
    >>> expr = 2 * (A + B)
    >>> with configure_printing(str_format='unicode'):
    ...     dot = dotprint(expr)
    >>> dot.strip() == r'''
    ... digraph{
    ...
    ... # Graph style
    ... "ordering"="out"
    ... "rankdir"="TD"
    ...
    ... #########
    ... # Nodes #
    ... #########
    ...
    ... "node_(0, 0)" ["label"="ScalarTimesOperator"];
    ... "node_(1, 0)" ["label"="2"];
    ... "node_(1, 1)" ["label"="OperatorPlus"];
    ... "node_(2, 0)" ["label"="Â⁽¹⁾"];
    ... "node_(2, 1)" ["label"="B̂⁽¹⁾"];
    ...
    ... #########
    ... # Edges #
    ... #########
    ...
    ... "node_(0, 0)" -> "node_(1, 0)"
    ... "node_(0, 0)" -> "node_(1, 1)"
    ... "node_(1, 1)" -> "node_(2, 0)"
    ... "node_(1, 1)" -> "node_(2, 1)"
    ... }'''.strip()
    True

The ``dot`` commandline program renders the code into an image:

.. figure:: ../_static/dotprint.svg

The various options of :func:`dotprint` allow for arbitrary customization of
the graph's structural and visual properties.

.. _DOT: http://www.graphviz.org
"""

from qnet.algebra.core.abstract_algebra import Expression, Operation
from collections import defaultdict

__all__ = []
__private__ = ['dotprint', 'expr_labelfunc']

template = r'''
digraph{

# Graph style
%(graphstyle)s

#########
# Nodes #
#########

%(nodes)s

#########
# Edges #
#########

%(edges)s
}
'''


def _attrprint(d, delimiter=', '):
    """Print a dictionary of attributes in the DOT format"""
    return delimiter.join(('"%s"="%s"' % item) for item in sorted(d.items()))


def _styleof(expr, styles):
    """Merge style dictionaries in order"""
    style = dict()
    for expr_filter, sty in styles:
        if expr_filter(expr):
            style.update(sty)
    return style


def _node_id(expr, location, idfunc, repeat=True):
    res = str(idfunc(expr))
    if repeat:
        res += '_%s' % str(location)
    return res


def expr_labelfunc(leaf_renderer=str, fallback=str):
    """Factory for function ``labelfunc(expr, is_leaf)``

    It has the following behavior:

    * If ``is_leaf`` is True, return ``leaf_renderer(expr)``.

    * Otherwise,

      - if `expr` is an Expression, return a custom string similar to
        :func:`~qnet.printing.srepr`, but with an ellipsis for ``args``
      - otherwise, return ``fallback(expr)``
    """

    def _labelfunc(expr, is_leaf):
        if is_leaf:
            label = leaf_renderer(expr)
        elif isinstance(expr, Expression):
            if len(expr.kwargs) == 0:
                label = expr.__class__.__name__
            else:
                label = "%s(..., %s)" % (
                    expr.__class__.__name__,
                    ", ".join([
                        "%s=%s" % (key, val)
                        for (key, val) in expr.kwargs.items()]))
        else:
            label = fallback(expr)
        return label

    return _labelfunc


def _op_children(expr):
    if isinstance(expr, Operation):
        return expr.operands
    else:
        return []


def dotprint(
        expr, styles=None, maxdepth=None, repeat=True,
        labelfunc=expr_labelfunc(str, str),
        idfunc=None, get_children=_op_children, **kwargs):
    """Return the `DOT`_ (graph) description of an Expression tree as a string

    Args:
        expr (object): The expression to render into a graph. Typically an
            instance of :class:`~qnet.algebra.abstract_algebra.Expression`, but
            with appropriate `get_children`, `labelfunc`, and `id_func`, this
            could be any tree-like object
        styles (list or None): A list of tuples ``(expr_filter, style_dict)``
            where ``expr_filter`` is a callable and ``style_dict`` is a list
            of `DOT`_ node properties that should be used when rendering a node
            for which ``expr_filter(expr)`` return True.
        maxdepth (int or None): The maximum depth of the resulting tree (any
            node at `maxdepth` will be drawn as a leaf)
        repeat (bool): By default, if identical sub-expressions occur in
            multiple locations (as identified by `idfunc`, they will be
            repeated in the graph. If ``repeat=False`` is given, each unique
            (sub-)expression is only drawn once.  The resulting graph may no
            longer be a proper tree, as recurring expressions will have
            multiple parents.
        labelfunc (callable): A function that receives `expr` and a boolean
            ``is_leaf`` and returns the label of the corresponding node in the
            graph. Defaults to ``expr_labelfunc(str, str)``.
        idfunc (callable or None): A function that returns the ID of the node
            representing a given expression. Expressions for which `idfunc`
            returns identical results are considered identical if `repeat` is
            False. The default value None uses a function that is appropriate
            to a single standalone DOT file. If this is insufficient, something
            like ``hash`` or ``str`` would make a good `idfunc`.
        get_children (callable): A function that return a list of
            sub-expressions (the children of `expr`). Defaults to the operands
            of an :class:`~qnet.algebra.abstract_algebra.Operation` (thus,
            anything that is not an Operation is a leaf)
        kwargs: All further keyword arguments set custom `DOT`_ graph
            attributes

    Returns:
        str: a multiline str representing a graph in the `DOT`_ language

    Notes:
        The node `styles` are additive. For example, consider the following
        custom styles::

            styles = [
                (lambda expr: isinstance(expr, SCALAR_TYPES),
                    {'color': 'blue', 'shape': 'box', 'fontsize': 12}),
                (lambda expr: isinstance(expr, Expression),
                    {'color': 'red', 'shape': 'box', 'fontsize': 12}),
                (lambda expr: isinstance(expr, Operation),
                    {'color': 'black', 'shape': 'ellipse'})]

        For Operations (which are a subclass of Expression) the color and shape
        are overwritten, while the fontsize 12 is inherited.

        Keyword arguments are directly translated into graph styles. For
        example, in order to produce a horizontal instead of vertical graph,
        use ``dotprint(..., rankdir='LR')``.

    See also:
        :func:`sympy.printing.dot.dotprint` provides an equivalent function for
        SymPy expressions.
    """
    # the routine is called 'dotprint' to match sympy (even though most of the
    # similar routines for the other printers are called e.g. 'latex', not
    # 'latexprint'
    if idfunc is None:
        if repeat:
            idfunc = lambda expr: 'node'
        else:
            idfunc = hash
    graphstyle = {'rankdir': 'TD', 'ordering': 'out'}
    graphstyle.update(kwargs)
    nodes = []
    edges = []
    level = 0
    pos = 0
    pos_counter = defaultdict(int)  # level => current pos
    stack = [(level, pos, expr)]
    if styles is None:
        styles = []
    while len(stack) > 0:
        level, pos, expr = stack.pop(0)
        node_id = _node_id(expr, (level, pos), idfunc, repeat)
        children = get_children(expr)
        is_leaf = len(children) == 0
        if maxdepth is not None and level >= maxdepth:
            is_leaf = True
        style = _styleof(expr, styles)
        style['label'] = labelfunc(expr, is_leaf)
        nodes.append('"%s" [%s];' % (node_id, _attrprint(style)))
        if not is_leaf:
            try:
                for expr_sub in children:
                    i_sub = pos_counter[level+1]
                    id_sub = _node_id(
                        expr_sub, (level+1, i_sub), idfunc, repeat)
                    edges.append('"%s" -> "%s"' % (node_id, id_sub))
                    stack.append((level+1, i_sub, expr_sub))
                    pos_counter[level+1] += 1
            except AttributeError:
                pass
    return template % {
        'graphstyle': _attrprint(graphstyle, delimiter='\n'),
        'nodes': '\n'.join(nodes),
        'edges': '\n'.join(edges)}
