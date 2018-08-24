"""Tools for working with equations"""

from sympy.core.sympify import SympifyError

from ...utils.unicode import grapheme_len, ljust, rjust
from ..core.abstract_algebra import substitute

__all__ = ['Eq']


class Eq():
    """Symbolic equation

    This class keeps track of the `lhs` and `rhs` of an equation across
    arbitrary manipulations

    Args:
        lhs (Expression): the left-hand-side of the equation
        rhs (Expression): the right-hand-side of the equation
        tag (None or str): a tag (equation number) to be shown when printing
            the equation

    Example:
        >>> ω, E0 = sympy.symbols('omega, E_0')
        >>> hbar = sympy.symbols('hbar', positive=True)
        >>> H_0, H_1 = (OperatorSymbol(s, hs=0) for s in ('H_0', 'H_1'))
        >>> H = OperatorSymbol('H', hs=0)
        >>> mu = OperatorSymbol('mu', hs=0)
        >>> eq0 = Eq(H_0, ω * Create(hs=0) * Destroy(hs=0) + E0, tag='0')
        >>> print(unicode(eq0, show_hs_label=False))
        Ĥ₀ = E₀ + ω â^† â    (0)
        >>> eq1 = Eq(H_1, mu + E0, tag='1')
        >>> print(unicode(eq1, show_hs_label=False))
        Ĥ₁ = E₀ + μ̂    (1)
        >>> eq = (
        ...     (eq0 + eq1).set_tag('2')
        ...     .apply_to_rhs(lambda expr: expr - 2*E0, cont=True)
        ...     .apply(lambda expr: expr * hbar, cont=True)
        ...     .apply_mtd_to_lhs(
        ...         'substitute', var_map={H_0 + H_1: H}, cont=True)
        ...     .apply(lambda expr: expr**2, cont=True)
        ...     .apply_mtd_to_rhs('substitute', var_map={mu: 0}, cont=True)
        ...     .apply_mtd_to_rhs('expand', cont=True, tag='⋆')
        ... )
        >>> print(unicode(eq, show_hs_label=False))
            Ĥ₀ + Ĥ₁ = 2 E₀ + μ̂ + ω â^† â                 (2)
                    = μ̂ + ω â^† â
        h̅ (Ĥ₀ + Ĥ₁) = h̅ (μ̂ + ω â^† â)
                h̅ Ĥ = h̅ (μ̂ + ω â^† â)
             h̅² Ĥ Ĥ = h̅² (μ̂ + ω â^† â) (μ̂ + ω â^† â)
                    = h̅² ω² â^† (𝟙 + â^† â) â
                    = h̅² ω² â^† â^† â â + h̅² ω² â^† â    (⋆)
        >>> (eq
        ...  .apply_mtd_to_lhs('substitute', eq.as_dict)
        ...  .verify().is_zero)
        True
    """

    def __init__(
            self, lhs, rhs, tag=None,
            _prev_lhs=None, _prev_rhs=None, _prev_tags=None):
        self._lhs = lhs
        self._prev_lhs = _prev_lhs or []
        self._prev_rhs = _prev_rhs or []
        self._prev_tags = _prev_tags or []
        self._rhs = rhs
        try:
            self._tag = int(tag)
        except (ValueError, TypeError):
            self._tag = tag

    @property
    def lhs(self):
        """The left-hand-side of the equation"""
        lhs = self._lhs
        i = 0
        while lhs is None:
            i -= 1
            lhs = self._prev_lhs[i]
        return lhs

    @property
    def rhs(self):
        """The right-hand-side of the equation"""
        return self._rhs

    @property
    def tag(self):
        """A tag (equation number) to be shown when printing the equation, or
        None"""
        return self._tag

    def set_tag(self, tag):
        """Return a copy of the equation with a new `tag`"""
        return Eq(
            self._lhs, self._rhs, tag=tag,
            _prev_lhs=self._prev_lhs, _prev_rhs=self._prev_rhs,
            _prev_tags=self._prev_tags)

    @property
    def as_dict(self):
        """Mapping of the lhs to the rhs

        This allows to plug an equation into another expression via
        :meth:`~.Expression.substitute`.
        """
        return {self.lhs: self.rhs}

    def apply(self, func, *args, cont=False, tag=None, **kwargs):
        """Apply `func` to both sides of the equation

        Returns a new equation where the left-hand-side and right-hand side
        are replaced by the application of `func`::

            lhs=func(lhs, *args, **kwargs)
            rhs=func(rhs, *args, **kwargs)

        If ``cont=True``, the resulting equation will keep a history of its
        previous state (resulting in multiple lines of equations when printed,
        as in the main example above).

        The resulting equation with have the given `tag`.
        """
        new_lhs = func(self.lhs, *args, **kwargs)
        if new_lhs == self.lhs and cont:
            new_lhs = None
        new_rhs = func(self.rhs, *args, **kwargs)
        new_tag = tag
        return self._update(new_lhs, new_rhs, new_tag, cont)

    def apply_to_lhs(self, func, *args, cont=False, tag=None, **kwargs):
        """Apply `func` to lhs of equation only

        Like :meth:`apply`, but modifying only the left-hand-side.
        """
        new_lhs = func(self.lhs, *args, **kwargs)
        new_rhs = self.rhs
        new_tag = tag
        return self._update(new_lhs, new_rhs, new_tag, cont)

    def apply_to_rhs(self, func, *args, cont=False, tag=None, **kwargs):
        """Apply `func` to rhs of equation only

        Like :meth:`apply`, but modifying only the right-hand-side.
        """
        if cont:
            new_lhs = None
        else:
            new_lhs = self.lhs
        new_rhs = func(self.rhs, *args, **kwargs)
        new_tag = tag
        return self._update(new_lhs, new_rhs, new_tag, cont)

    def apply_mtd(self, mtd, *args, cont=False, tag=None, **kwargs):
        """Call the method `mtd` on both sides of the equation

        That is, the left-hand-side and right-hand-side are replaced by::

            lhs=lhs.<mtd>(*args, **kwargs)
            rhs=rhs.<mtd>(*args, **kwargs)

        The `cont` and `tag` parameters are as in :meth:`apply`.
        """
        new_lhs = getattr(self.lhs, mtd)(*args, **kwargs)
        if new_lhs == self.lhs and cont:
            new_lhs = None
        new_rhs = getattr(self.rhs, mtd)(*args, **kwargs)
        new_tag = tag
        return self._update(new_lhs, new_rhs, new_tag, cont)

    def apply_mtd_to_lhs(self, mtd, *args, cont=False, tag=None, **kwargs):
        """Call the method `mtd` on the lhs of the equation only.

        Like :meth:`apply_mtd`, but modifying only the left-hand-side.
        """
        new_lhs = getattr(self.lhs, mtd)(*args, **kwargs)
        new_rhs = self.rhs
        new_tag = tag
        return self._update(new_lhs, new_rhs, new_tag, cont)

    def apply_mtd_to_rhs(self, mtd, *args, cont=False, tag=None, **kwargs):
        """Call the method `mtd` on the rhs of the equation

        Like :meth:`apply_mtd`, but modifying only the right-hand-side.
        """
        new_lhs = self.lhs
        if cont:
            new_lhs = None
        new_rhs = getattr(self.rhs, mtd)(*args, **kwargs)
        new_tag = tag
        return self._update(new_lhs, new_rhs, new_tag, cont)

    def substitute(self, var_map, cont=False, tag=None):
        """Substitute sub-expressions both on the lhs and rhs

        Args:
            var_map (dict): Dictionary with entries of the form
                ``{expr: substitution}``
        """
        return self.apply(substitute, var_map=var_map, cont=cont, tag=tag)

    def _update(self, new_lhs, new_rhs, new_tag, cont):
        if not cont:
            new_prev_lhs = None
            new_prev_rhs = None
            new_prev_tags = None
        else:
            new_prev_lhs = self._prev_lhs.copy()
            new_prev_lhs.append(self._lhs)
            new_prev_rhs = self._prev_rhs.copy()
            new_prev_rhs.append(self.rhs)
            new_prev_tags = self._prev_tags.copy()
            new_prev_tags.append(self.tag)
        return Eq(
            new_lhs, new_rhs, tag=new_tag,
            _prev_lhs=new_prev_lhs, _prev_rhs=new_prev_rhs,
            _prev_tags=new_prev_tags)

    def verify(self, func=None, *args, **kwargs):
        """Subtract the rhs from the lhs of the equation

        Before the substraction, each side is expanded and any scalars are
        simplified. If given, `func` with the positional arguments `args` and
        keyword-arguments `kwargs` is applied to the result before returning
        it.

        You may complete the verification by checking the :attr:`is_zero`
        attribute of the returned expression.
        """
        res = (
            self.lhs.expand().simplify_scalar() -
            self.rhs.expand().simplify_scalar())
        if func is not None:
            return func(res, *args, **kwargs)
        else:
            return res

    def copy(self):
        """Return a copy of the equation"""
        return Eq(
            self._lhs, self._rhs, tag=self._tag,
            _prev_lhs=self._prev_lhs, _prev_rhs=self._prev_rhs,
            _prev_tags=self._prev_tags)

    @property
    def free_symbols(self):
        """Set of free SymPy symbols contained within the equation."""
        try:
            lhs_syms = self.lhs.free_symbols
        except AttributeError:
            lhs_syms = set()
        try:
            rhs_syms = self.rhs.free_symbols
        except AttributeError:
            rhs_syms = set()
        return lhs_syms | rhs_syms

    @property
    def bound_symbols(self):
        """Set of bound SymPy symbols contained within the equation."""
        try:
            lhs_syms = self.lhs.bound_symbols
        except AttributeError:
            lhs_syms = set()
        try:
            rhs_syms = self.rhs.bound_symbols
        except AttributeError:
            rhs_syms = set()
        return lhs_syms | rhs_syms

    @property
    def all_symbols(self):
        """Combination of :attr:`free_symbols` and :attr:`bound_symbols`"""
        return self.free_symbols | self.bound_symbols

    def __add__(self, other):
        try:
            return Eq(lhs=(self.lhs + other.lhs), rhs=(self.rhs + other.rhs))
        except AttributeError:
            return Eq(lhs=(self.lhs + other), rhs=(self.rhs + other))

    __radd__ = __add__

    def __sub__(self, other):
        try:
            return Eq(lhs=(self.lhs - other.lhs), rhs=(self.rhs - other.rhs))
        except AttributeError:
            return Eq(lhs=(self.lhs - other), rhs=(self.rhs - other))

    def __mul__(self, other):
            return Eq(lhs=(self.lhs * other), rhs=(self.rhs * other))

    def __rmul__(self, other):
            return Eq(lhs=(other * self.lhs), rhs=(other * self.rhs))

    def __truediv__(self, other):
            return Eq(lhs=(self.lhs / other), rhs=(self.rhs / other))

    def __eq__(self, other):
        try:
            return self.lhs == other.lhs and self.rhs == other.rhs
        except AttributeError:
            return self.rhs == other

    def _render_str(self, renderer, *args, **kwargs):
        rendered_lhs = []
        rendered_rhs = []
        rendered_tags = []

        for i, rhs in enumerate(self._prev_rhs):
            lhs = self._prev_lhs[i]
            tag = self._prev_tags[i]
            if lhs is None:
                rendered_lhs.append('')
            else:
                rendered_lhs.append(renderer(lhs, *args, **kwargs))
            rendered_rhs.append(renderer(rhs, *args, **kwargs))
            if tag is None:
                rendered_tags.append('')
            else:
                rendered_tags.append(renderer(tag, *args, **kwargs))
        if self._lhs is None:
            rendered_lhs.append('')
        else:
            rendered_lhs.append(renderer(self._lhs, *args, **kwargs))
        rendered_rhs.append(renderer(self._rhs, *args, **kwargs))
        if self._tag is None:
            rendered_tags.append('')
        else:
            rendered_tags.append(renderer(self._tag, *args, **kwargs))
        len_lhs = max([grapheme_len(s) for s in rendered_lhs])
        len_rhs = max([grapheme_len(s) for s in rendered_rhs])
        len_tag = max([grapheme_len(s) for s in rendered_tags]) + 2

        lines = []
        for (lhs, rhs, tag) in zip(rendered_lhs, rendered_rhs, rendered_tags):
            if len(tag) > 0:
                tag = "(" + tag + ")"
            lhs = rjust(lhs, len_lhs)
            rhs = ljust(rhs, len_rhs)
            tag = ljust(tag, len_tag)
            lines.append((lhs + ' = ' + rhs + "    " + tag).rstrip())
        return "\n".join(lines)

    def __str__(self):
        return self._render_str(renderer=str)

    def __repr__(self):
        return self._render_str(renderer=repr)

    def _repr_latex_(self):
        from qnet.printing import latex
        return latex(self)

    def _sympy_(self):
        raise SympifyError("QNET Eq cannot be converted to SymPy")
