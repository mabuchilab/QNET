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
"""Provides the base class for Printers"""

from typing import Any, Tuple
from abc import ABCMeta
from collections import OrderedDict

from numpy import complex128
import sympy

from ..algebra.singleton import Singleton
from ..algebra.scalar_types import SCALAR_TYPES

class Printer(metaclass=ABCMeta):
    """Base class for Printers (and default ASCII printer)

    Attributes:
        head_repr_fmt (str): The format for representing expressions in the
            form ``head(arg1, arg2, ..., key1=val1, key2=val2, ...)``. Uses
            formatting keys `head` (``expr.__class__.__name__``), `args`
            (rendered representation of ``expr.args``), and `kwargs` (rendered
            representation of ``expr.kwargs``). Used by
            :meth:`render_head_repr`
        identity_sym (str): Representation of the identity operator
        circuit_identify_fmt (str): Format for the identity in a Circuit,
            parametrized by the number of channels, given as the formatting key
            `cdim`.
        dagger_sym (str): Symbol representing a dagger
        daggered_sym (str): Superscript version of `dagger_sym`
        permutation_sym (str): The identifier of a Circuit permutation
        pseudo_daggered_sym (str): Superscript representing a pseudo-dagger
        pal_left (str): The symbol/string for a left parenthesis
        par_right (str): The symbol/string for a right parenthesis
        brak_left (str): The symbol/string for a left square bracket
        brak_right (str): The symbol/string for a right square bracket
        arg_sep (str): The string that should be used to separate rendered
            arguments in a list (usually a comma)
        scalar_product_sym (str): Symbol to indicate a product between two
            scalars
        tensor_sym (str): Symbol to indicate a tensor product
        inner_product_sym (str): Symbol to indicate an inner product
        op_product_sym (str): Symbol to indicate a product between two
            operators in the the same Hilbert space
        circuit_series (str): Infix symbol for a series product of two
            circuits
        circuit_concat_sym (str): Infix symbol for a concatenation of two
            circuits
        circuit_inverse_fmt (str): Format for rendering the series-inverse of a
            circuit element. Receives a formatting key `operand` of the
            rendered operand circuit element
        circuit_fb_fmt (str): Format for rendering a feedback circuit element.
            Receives the formatting keys `operand`, `output`, and `input` that
            are the rendered operand circuit element, the index of the ouput
            port (as a string), and the index of the input port to which the
            feedback connects (also as a string)
        op_trace_fmt (str): Format for rendering a trace. Receives the
            formatting keys `operand` (the object being traced) and `space`
            (the rendered label of the Hilbert space that is being traced over)
        null_space_proj_sym (str): The identifier for a nullspace projector
        hilbert_space_fmt (str): Format for rendering a :class:`HilbertSpace`
            object. Receives the formatting key `label` with the rendered label
            of the Hilbert space
        matrix_left_sym (str): The symbol that marks the beginning of a matrix,
            for rendering a :class:`Matrix` instance
        matrix_right_sym (str): The symbol that marks the end of a matrix
        matrix_row_left_sym (str): Symbol that marks beginning of row in matrix
        matrix_row_right_sym (str): Symbol that marks end of row in matrix
        matrix_col_sep_sym (str): Symbol that separates the values in different
            columns of a matrix
        matrix_row_sep_sym (str): Symbol that separates the rows of a matrix
        bra_fmt (str): Format for rendering a :class:`Bra` instance. Receives
            the formatting keys `label` (the rendered label of the state)
            and `space` (the rendered label of the Hilbert space)
        ket_fmt (str): Format for rendering a :class:`Ket` instance. Receives
            the formatting keys `label` and `space`
        ketbra_fmt (str): Format for rendering a :class:`KetBra` instance.
            Receives the formatting keys `label_i`, `label_j`, and `space`, for
            the rendered label of the "left" and "right" state, and the Hilbert
            space
        braket_fmt (str): Format for rendering a :class:`BraKet` instance.
            Receives the formatting keys `label_i`, `label_j`, and `space`.
        commut_fmt (str): format for rendering a commutator. Receives the
            formatting keys `A` and `B`. Used by :meth:`render_commutator`.
        anti_commut_fmt (str): format for rendering an anti-commutator.
            Receives the formatting keys `A` and `B`. Used by
            :meth:`render_commutator`.
        cc_string (str): String to indicate the complex conjugate (in a sum)
    """

    head_repr_fmt = r'{head}({args}{kwargs})'
    identity_sym = '1'
    circuit_identity_fmt = r'cid({cdim})'
    zero_sym = '0'
    dagger_sym = r'H'
    daggered_sym = r'^H'
    permutation_sym = 'Perm'
    pseudo_daggered_sym = r'^+'
    par_left = '('
    par_right = ')'
    brak_left = '['
    brak_right = ']'
    arg_sep = ", "
    scalar_product_sym = r'*'
    tensor_sym = r'*'
    inner_product_sym = r'*'
    op_product_sym = r'*'
    circuit_series_sym = '<<'
    circuit_concat_sym = '+'
    circuit_inverse_fmt = r'[{operand}]^{{-1}}'
    circuit_fb_fmt = r'[{operand}]_{{{output}->{input}}}'
    op_trace_fmt = r'tr_({space})[{operand}]'
    null_space_proj_sym = "P_Ker"
    hilbert_space_fmt = r'H_{label}'
    matrix_left_sym = '['
    matrix_right_sym = ']'
    matrix_row_left_sym = '['
    matrix_row_right_sym = ']'
    matrix_col_sep_sym = ', '
    matrix_row_sep_sym = ', '
    bra_fmt = r'<{label}|_({space})'
    ket_fmt = r'|{label}>_({space})'
    ketbra_fmt = r'|{label_i}><{label_j}|_({space})'
    braket_fmt = r'<{label_i}|{label_j}>_({space})'
    commut_fmt = r'[{A}, {B}]'
    anti_commut_fmt = r'{{{A}, {B}}}'
    cc_string = r'c.c.'

    op_hs_super_sub = 1  # where to put Hilbert space label for operators

    _special_render = [
        (SCALAR_TYPES, 'render_scalar'),
        #(str, '_render_rendered'),
    ]

    _registry = {}
    # Note: do NOT set Printer._registry = {} instead of using clear_registry!
    # This would create an instance attribute that shadows the class attribute

    @classmethod
    def render(cls, expr: Any, adjoint=False) -> str:
        """Render an expression (or the adjoint of the expression)"""
        # try to return known expression from _registry
        try:
            if not adjoint and expr in cls._registry:
                return cls._registry[expr]
        except TypeError:
            pass  # unhashable types, e.g. numpy array
        # handle special classes
        for special_cls, special_render in cls._special_render:
            if isinstance(expr, special_cls):
                mtd = getattr(cls, special_render)
                res = mtd(expr, adjoint=adjoint)
                if isinstance(res, str):
                    return res
        try:
            return cls._render(expr, adjoint=adjoint)
        # fall back
        except AttributeError:
            return cls._fallback(expr, adjoint=adjoint)

    @classmethod
    def _render(cls, expr, adjoint=False):
        """Render the expression (usually by delegating to a method of expr),
        or throw an AttributeError"""
        return expr._ascii_(adjoint=adjoint)

    @classmethod
    def _fallback(cls, expr, adjoint=False):
        """Render an expression that does not have _delegate_mtd"""
        if adjoint:
            return "Adjoint[%s]" % str(expr)
        else:
            return str(expr)

    @classmethod
    def _render_rendered(cls, expr, adjoint=False):
        """Render an already rendered string"""
        assert not adjoint, \
            "Cannot render the adjoint of an already rendered expression"
        return expr

    @classmethod
    def register(cls, expr, rendered):
        """Register a fixed rendered string for the given `expr` in an internal
        registry. As a result, any call to :meth:`render`
        for `expr` will immediately return `rendered`"""
        cls._registry[expr] = rendered

    @classmethod
    def update_registry(cls, mapping):
        """Call ``register(key, val)`` for every key-value pair in the
        `mapping dictionary`"""
        cls._registry.update(mapping)

    @classmethod
    def del_registered_expr(cls, expr):
        """Remove the registered `expr` from the registry (cf.
        :meth:`register_expr`)"""
        del cls._registry[expr]

    @classmethod
    def clear_registry(cls):
        """Clear the registry"""
        cls._registry = {}

    @classmethod
    def render_head_repr(
            cls, expr: Any, sub_render=None, key_sub_render=None) -> str:
        """Render a textual representation of `expr` using
        :attr:`head_repr_fmt`. Positional and keyword arguments are recursively
        rendered using `sub_render`, which defaults to `cls.render` by default.
        If desired, a different renderer may be used for keyword arguments by
        giving `key_sub_renderer`

        Raises:
            AttributeError: if `expr` is not an instance of
                :class:`Expression`, or more specifically, if `expr` does not
                have `args` and `kwargs` (respectively `minimal_kwargs`)
                properties
        """
        if sub_render is None:
            sub_render = cls.render
        if key_sub_render is None:
            key_sub_render = sub_render
        if isinstance(expr.__class__, Singleton):
            # We exploit that Singletons override __expr__ to directly return
            # their name
            return repr(expr)
        args = expr.args
        keys = expr.minimal_kwargs.keys()
        kwargs = ''
        if len(keys) > 0:
            kwargs = cls.arg_sep.join(
                        ["%s=%s" % (key, key_sub_render(expr.kwargs[key]))
                         for key in keys])
            if len(args) > 0:
                kwargs = cls.arg_sep + kwargs
        return cls.head_repr_fmt.format(
                head=expr.__class__.__name__,
                args=cls.arg_sep.join([sub_render(arg) for arg in args]),
                kwargs=kwargs)

    @classmethod
    def _split_identifier(cls, identifier:str) -> Tuple[str, str]:
        """Split the given identifier at the first underscore into (rendered)
        name and subscript. Both `name` and `subscript` are rendered through
        the `render_string` method"""
        try:
            name, subscript = identifier.split("_", 1)
        except (TypeError, ValueError):
            name = identifier
            subscript = ''
        return cls.render_string(name), cls.render_string(subscript)

    @classmethod
    def _split_op(
        cls, identifier: str, hs_label=None, dagger=False, args=None) \
            -> Tuple[str, str, str, str]:
        """Return `name`, total `subscript`, total `superscript` and
        `arguments` str. All of the returned strings are fully rendered.

        Args:
        identifier (str): An (non-rendered/ascii) identifier that may include a
            subscript
        hs_label (str): The rendered label for the Hilbert space of the
            operator, or None
        dagger (bool): Flag to indicate whether the operator is daggered
        args (list or None): List of arguments (expressions), which will be
            rendered, joined with `arg_sep`, and eclosed in parentheses. If no
            arguments should be rendered, None.
        """
        name, total_subscript = cls._split_identifier(identifier)
        total_superscript = ''
        if (hs_label is not None and hs_label != '' and
                cls.op_hs_super_sub != 0):
            if cls.op_hs_super_sub > 0:
                total_superscript += '(' + hs_label + ')'
            else:
                if len(total_subscript) == 0:
                    total_subscript = '(' + hs_label + ')'
                else:
                    total_subscript += ',(' + hs_label + ')'
        if dagger:
            total_superscript += cls.dagger_sym
        args_str = ''
        if (args is not None) and (len(args) > 0):
            args_str = (cls.par_left +
                        cls.arg_sep.join([cls.render(arg) for arg in args]) +
                        cls.par_right)
        return name, total_subscript, total_superscript, args_str

    @classmethod
    def render_op(cls, identifier: str, hs=None, dagger=False, args=None,
                  superop=False) -> str:
        """Render an operator

        Args:
            identifier(str): Name of the operator (unrendered string)
            hs (HilbertSpace): Hilbert space instance of the operator
            dagger (bool): Whether or not to render the operator with a dagger
            args  (list): List of arguments for the operator (list of
                expressions). These will be rendered through the `render`
                method and appended to the rendered operator in parentheses
            superop (bool): Flag to indicate whether the operator is a
                 superoperator
        """
        hs_label = None
        if hs is not None:
            hs_label = cls.render_hs_label(hs)
        name, total_subscript, total_superscript, args_str \
            = cls._split_op(identifier, hs_label, dagger, args)
        res = name
        if len(total_subscript) > 0:
            res += "_" + total_subscript
        if len(total_superscript) > 0:
            res += "^" + total_superscript
        if len(args_str) > 0:
            res += args_str
        return res

    @classmethod
    def render_string(cls, ascii_str: str) -> str:
        """Render an unrendered (ascii) string, resolving e.g. greek letters
        and sub-/superscripts"""
        return ascii_str

    @classmethod
    def render_sum(
            cls, operands, plus_sym='+', minus_sym='-', padding=' ',
            adjoint=False, lower_classes=()):
        """Render a sum"""
        parts = []
        if len(plus_sym.strip()) > 0:
            padded_plus_sym = padding + plus_sym.strip() + padding
        padded_minus_sym = padding + minus_sym.strip() + padding
        for i_op, operand in enumerate(operands):
            part = cls.render(operand, adjoint=adjoint).strip()
            if isinstance(operand, lower_classes):
                part = cls.par_left + part + cls.par_right
            if i_op > 0:
                if part.startswith(minus_sym):
                    parts.append(padded_minus_sym)
                    part = part[len(minus_sym):].strip()
                else:
                    parts.append(padded_plus_sym)
            parts.append(part)
        return "".join(parts).strip()

    @classmethod
    def render_product(
            cls, operands, prod_sym, sum_classes, minus_sym='-', padding=' ',
            adjoint=False, dynamic_prod_sym=None):
        """Render a product"""
        parts = []
        if len(prod_sym.strip()) > 0:
            padded_prod_sym = padding + prod_sym.strip() + padding
        else:
            padded_prod_sym = prod_sym
        prev_op = None
        for i_op, operand in enumerate(operands):
            part = cls.render(operand, adjoint=adjoint).strip()
            if isinstance(operand, sum_classes):
                part = cls.par_left + part + cls.par_right
            elif part.startswith(minus_sym) and i_op > 0:
                part = cls.par_left + part + cls.par_right
            if i_op > 0:
                if dynamic_prod_sym is not None:
                    prod_sym = dynamic_prod_sym(prev_op, operand)
                    if len(prod_sym.strip()) > 0:
                        padded_prod_sym = padding + prod_sym.strip() + padding
                    else:
                        padded_prod_sym = prod_sym
                parts.append(padded_prod_sym)
            parts.append(part)
            prev_op = operand
        return "".join(parts).strip()


    @classmethod
    def render_hs_label(cls, hs: Any) -> str:
        """Render the total label for the given Hilbert space"""
        if isinstance(hs.__class__, Singleton):
            return cls.render_string(hs.label)
        else:
            return cls.render_product(
                    [cls.render_string(ls.label) for ls in hs.local_factors],
                    prod_sym=cls.tensor_sym, sum_classes=(), padding='')

    @classmethod
    def render_scalar(cls, value: Any, adjoint=False) -> str:
        """Render a scalar `value` (numeric or symbolic)"""
        if adjoint:
            res = str(sympy.conjugate(value))
        else:
            res = str(value)
        return res

    @classmethod
    def render_commutator(cls, A, B, adjoint=False, anti_commutator=False):
        """Render a commutator or anti-commutator"""
        A_rendered = cls.render(A)
        B_rendered = cls.render(B)
        if anti_commutator:
            res = cls.anti_commut_fmt.format(A=A_rendered, B=B_rendered)
        else:
            res = cls.commut_fmt.format(A=A_rendered, B=B_rendered)
        if adjoint:
            return res + cls.daggered_sym
        else:
            return res
