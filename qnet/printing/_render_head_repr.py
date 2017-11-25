# This file is part of QNET.
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

from typing import Any

from sympy.printing.repr import srepr as sympy_srepr
from sympy.core.basic import Basic as SympyBasic

from ..algebra.singleton import Singleton
from ..algebra.abstract_algebra import Expression


def render_head_repr(
        expr: Any, sub_render=None, key_sub_render=None) -> str:
    """Render a textual representation of `expr` using
    Positional and keyword arguments are recursively
    rendered using `sub_render`, which defaults to `render_head_repr` by
    default.  If desired, a different renderer may be used for keyword
    arguments by giving `key_sub_renderer`

    Raises:
        AttributeError: if `expr` is not an instance of
            :class:`Expression`, or more specifically, if `expr` does not
            have `args` and `kwargs` (respectively `minimal_kwargs`)
            properties
    """
    head_repr_fmt = r'{head}({args}{kwargs})'
    if sub_render is None:
        sub_render = render_head_repr
    if key_sub_render is None:
        key_sub_render = sub_render
    if isinstance(expr.__class__, Singleton):
        # We exploit that Singletons override __expr__ to directly return
        # their name
        return repr(expr)
    if isinstance(expr, Expression):
        args = expr.args
        keys = expr.minimal_kwargs.keys()
        kwargs = ''
        if len(keys) > 0:
            kwargs = ", ".join(
                        ["%s=%s" % (key, key_sub_render(expr.kwargs[key]))
                            for key in keys])
            if len(args) > 0:
                kwargs = ", " + kwargs
        return head_repr_fmt.format(
            head=expr.__class__.__name__,
            args=", ".join([sub_render(arg) for arg in args]),
            kwargs=kwargs)
    elif isinstance(expr, SympyBasic):
        return sympy_srepr(expr)
    else:
        return repr(expr)
