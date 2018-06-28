from typing import Any

from sympy.printing.repr import srepr as sympy_srepr
from sympy.core.basic import Basic as SympyBasic

from ..utils.singleton import Singleton
from qnet.algebra.core.abstract_algebra import Expression


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
            kwargs = ", ".join([
                "%s=%s" % (key, key_sub_render(expr.kwargs[key]))
                for key in keys])
            if len(args) > 0:
                kwargs = ", " + kwargs
        return head_repr_fmt.format(
            head=expr.__class__.__name__,
            args=", ".join([sub_render(arg) for arg in args]),
            kwargs=kwargs)
    elif isinstance(expr, (tuple, list)):
        delims = ("(", ")") if isinstance(expr, tuple) else ("[", "]")
        if len(expr) == 1:
            delims = (delims[0], "," + delims[1])
        return (
            delims[0] +
            ", ".join([
                render_head_repr(
                    v, sub_render=sub_render, key_sub_render=key_sub_render)
                for v in expr]) +
            delims[1])
    else:
        return sympy_srepr(expr)
