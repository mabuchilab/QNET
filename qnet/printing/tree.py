#    This file is part of QNET.
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
# Copyright (C) 2012-2017, QNET authors (see AUTHORS file)
#
###########################################################################
"""Tree printer for Expressions

This is mainly for interactive use.
"""
from ._render_head_repr import render_head_repr

__all__ = ['tree', 'print_tree']


def _shorten_render(renderer, max_len):
    """Return a modified that returns the representation of expr, or '...' if
    that representation is longer than `max_len`"""

    def short_renderer(expr):
        res = renderer(expr)
        if len(res) > max_len:
            return '...'
        else:
            return res

    return short_renderer


def _shorten_render_unicode():
    from qnet.printing import unicode as unicode_printer
    return _shorten_render(unicode_printer, 15)


def _shorten_render_ascii():
    from qnet.printing import ascii as ascii_printer
    return _shorten_render(ascii_printer, 15)


def print_tree(
        expr, attr='operands', padding='', exclude_type=None, depth=None,
        unicode=True, srepr_leaves=False, _last=False, _root=True, _level=0,
        _print=True):
    """Print a tree representation of the structure of `expr`

    Args:
        expr (Expression): expression to render
        attr (str): The attribute from which to get the children of `expr`
        padding (str): Whitespace by which the entire tree is idented
        exclude_type (type): Type (or list of types) which should never be
            expanded recursively
        depth (int or None): Maximum depth of the tree to be printed
        unicode (bool): If True, use unicode line-drawing symbols for the tree,
            and print expressions in a unicode representation.
            If False, use an ASCII approximation.
        srepr_leaves (bool): Whether or not to render leaves with `srepr`,
            instead of `ascii`/`unicode`

    See also:
        :func:`tree` return the result as a string, instead of printing it
    """
    from qnet.printing import srepr
    lines = []
    if unicode:
        draw = {'leaf': '└─ ', 'branch': '├─ ', 'line': '│'}
        sub_render = _shorten_render_unicode()
    else:
        draw = {'leaf': '+- ', 'branch': '+- ', 'line': '|'}
        sub_render = _shorten_render_ascii()
    to_str = lambda expr: render_head_repr(
            expr, sub_render=sub_render, key_sub_render=sub_render)
    if _root:
        lines.append(". " + to_str(expr))
    else:
        if _last:
            lines.append(padding[:-1] + draw['leaf'] + to_str(expr))
        else:
            lines.append(padding[:-1] + draw['branch'] + to_str(expr))
    padding = padding + '  '
    try:
        children = getattr(expr, attr)
    except AttributeError:
        children = []
    if exclude_type is not None:
        if isinstance(expr, exclude_type):
            children = []
    if depth is not None:
        if depth <= _level:
            children = []
    for count, child in enumerate(children):
        if hasattr(child, attr):
            if count == len(children)-1:
                lines += print_tree(
                    child, attr, padding + ' ',
                    exclude_type=exclude_type, depth=depth, unicode=unicode,
                    srepr_leaves=srepr_leaves, _last=True, _root=False,
                    _level=_level+1)
            else:
                lines += print_tree(
                    child, attr, padding + draw['line'],
                    exclude_type=exclude_type, depth=depth, unicode=unicode,
                    srepr_leaves=srepr_leaves, _last=False, _root=False,
                    _level=_level+1)
        else:
            if count == len(children)-1:
                if srepr_leaves:
                    lines.append(padding + draw['leaf'] + srepr(child))
                else:
                    lines.append(padding + draw['leaf'] + to_str(child))
            else:
                if srepr_leaves:
                    lines.append(padding + draw['branch'] + srepr(child))
                else:
                    lines.append(padding + draw['branch'] + to_str(child))
    if _root:
        if _print:
            print("\n".join(lines))
        else:
            return lines
    else:
        return lines


def tree(expr, **kwargs):
    """Give the output of `tree` as a multiline string, using line drawings to
    visualize the hierarchy of expressions (similar to the ``tree`` unix
    command line program for showing directory trees)

    See also:
        :func:`qnet.printing.srepr` with ``indented=True`` produces a similar
        tree-like rendering of the given expression that can be re-evaluated to
        the original expression.
    """
    return "\n".join(print_tree(expr, _print=False, **kwargs))
