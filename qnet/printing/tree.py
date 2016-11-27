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
# Copyright (C) 2016, Michael Goerz
#
###########################################################################
"""Tree printer for Expressions"""

from .base import Printer
from .unicode import UnicodePrinter
from .ascii import AsciiPrinter
from ..algebra.singleton import Singleton, singleton_object


@singleton_object
class HeadStrPrinter(Printer, metaclass=Singleton):
    """Printer that renders all expressions to in a "head" format, but using
    `sub_render` for the components of the Expression"""

    sub_render = str

    @classmethod
    def render(cls, expr, adjoint=False):
        """Render an expression"""
        if adjoint:
            raise NotImplementedError("adjoint not implemented")
        try:
            return cls.render_head_repr(expr, sub_render=cls.sub_render)
        except AttributeError:
            return str(expr)


def tree(expr, attr='operands', padding='', to_str=HeadStrPrinter.render,
         exclude_type=None, depth=None, unicode=True,
         _last=False, _root=True, _level=0, _print=True):
    """Print a tree representation of the structure of `expr`

    Args:
        expr (Expression): expression to render
        attr (str): The attribute from which to get the children of `expr`
        padding (str): Whitespace by which the entire tree is idented
        to_str (callable): Renderer for `expr`
        exclude_type (type): Type (or list of types) which should never be
            expanded recursively
        depth (int or None): Maximum depth of the tree to be printed
        unicode (bool): If True, use unicode line-drawing symbols for the tree.
            If False, use an ASCII approximation

    See also:
        `tree_str` return the result as a string, instead of printing it
    """
    lines = []
    if unicode:
        draw = {'leaf': '└─ ', 'branch': '├─ ', 'line': '│'}
        HeadStrPrinter.__class__.sub_render = UnicodePrinter.render
    else:
        draw = {'leaf': '+- ', 'branch': '+- ', 'line': '|'}
        HeadStrPrinter.__class__.sub_render = AsciiPrinter.render
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
                lines += tree(child, attr, padding + ' ', to_str,
                              exclude_type=exclude_type, depth=depth,
                              unicode=unicode, _last=True, _root=False,
                              _level=_level+1)
            else:
                lines += tree(child, attr, padding + draw['line'], to_str,
                              exclude_type=exclude_type, depth=depth,
                              unicode=unicode, _last=False, _root=False,
                              _level=_level+1)
        else:
            if count == len(children)-1:
                lines.append(padding + draw['leaf'] + to_str(child))
            else:
                lines.append(padding + draw['branch'] + to_str(child))
    if _root:
        if _print:
            print("\n".join(lines))
        else:
            return lines
    else:
        return lines


def tree_str(expr, **kwargs):
    """Give the output of `tree` as a multiline string"""
    return "\n".join(tree(expr, _print=False, **kwargs))
