"""Collection of operators that act on a bosonic Fock space"""
import re
from abc import ABCMeta
from collections import OrderedDict

from ..core.hilbert_space_algebra import LocalSpace
from ..core.algebraic_properties import implied_local_space, match_replace
from ..core.operator_algebra import LocalOperator, PseudoInverse
from ...utils.properties_for_args import properties_for_args

__all__ = [
    'Destroy', 'Create', 'Phase', 'Displace', 'Squeeze']


class Destroy(LocalOperator):
    """Bosonic annihilation operator

    It obeys the bosonic commutation relation::

        >>> Destroy(hs=1) * Create(hs=1) - Create(hs=1) * Destroy(hs=1)
        IdentityOperator
        >>> Destroy(hs=1) * Create(hs=2) - Create(hs=2) * Destroy(hs=1)
        ZeroOperator
    """

    _identifier = 'a'
    _dagger = False
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*$')

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    def _adjoint(self):
        return Create(hs=self.space)

    def _pseudo_inverse(self):
        return PseudoInverse(self)

    @property
    def identifier(self):
        """The identifier (symbol) that is used when printing the annihilation
        operator. This is identical to the identifier of :class:`Create`. A
        custom identifier for both :class:`Destroy` and :class:`Create` can be
        set through the `local_identifiers` parameter of the associated Hilbert
        space::

            >>> hs_custom = LocalSpace(0, local_identifiers={'Destroy': 'b'})
            >>> Create(hs=hs_custom).identifier
            'b'
            >>> Destroy(hs=hs_custom).identifier
            'b'
        """
        identifier = self._hs._local_identifiers.get(
            self.__class__.__name__, self._hs._local_identifiers.get(
                'Create', self._identifier))
        if not self._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'"
                % (identifier, self._rx_identifier.pattern))
        return identifier


class Create(LocalOperator):
    """Bosonic creation operator

    This is the adjoint of :class:`Destroy`.
    """

    _identifier = 'a'
    _dagger = True
    _rx_identifier = re.compile('^[A-Za-z][A-Za-z0-9]*$')

    def __init__(self, *, hs):
        super().__init__(hs=hs)

    def _adjoint(self):
        return Destroy(hs=self.space)

    def _pseudo_inverse(self):
        return PseudoInverse(self)

    @property
    def identifier(self):
        """The identifier (symbols) that is used when printing the creation
        operator. This is identical to the identifier of :class:`Destroy`"""
        identifier = self._hs._local_identifiers.get(
            self.__class__.__name__, self._hs._local_identifiers.get(
                'Destroy', self._identifier))
        if not self._rx_identifier.match(identifier):
            raise ValueError(
                "identifier '%s' does not match pattern '%s'"
                % (identifier, self._rx_identifier.pattern))
        return identifier


@properties_for_args
class Phase(LocalOperator):
    r"""Unitary "phase" operator

    .. math::

        \op{P}_{\rm hs}(\phi) =
        \exp\left(i \phi \Op{a}_{\rm hs}^\dagger \Op{a}_{\rm hs}\right)

    where :math:`a_{\rm hs}` is the annihilation operator acting on the
    :class:`.LocalSpace` `hs`.

    Args:
        phase (Scalar): the phase $\phi$
        hs (HilbertSpace or int or str): The Hilbert space on which the
            operator acts

    Printers should represent this operator with the default identifier::

        >>> Phase._identifier
        'Phase'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'Phase'
    _arg_names = ('phase', )
    _rules = OrderedDict()
    simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def _adjoint(self):
        return Phase.create(-self.phase.conjugate(), hs=self.space)

    def _pseudo_inverse(self):
        return Phase.create(-self.phase, hs=self.space)


@properties_for_args
class Displace(LocalOperator):
    r"""Unitary coherent displacement operator

    .. math::

        \op{D}_{\rm hs}(\alpha) =
        \exp\left({\alpha \Op{a}_{\rm hs}^\dagger -
                   \alpha^* \Op{a}_{\rm hs}}\right)

    where :math:`\Op{a}_{\rm hs}` is the annihilation operator acting on the
    :class:`.LocalSpace` `hs`.

    Args:
        displacement (Scalar): the displacement amplitude $\alpha$
        hs (HilbertSpace or int or str): The Hilbert space on which the
            operator acts

    Printers should represent this operator with the default identifier::

        >>> Displace._identifier
        'D'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = 'D'
    _nargs = 1
    _arg_names = ('displacement', )
    _rules = OrderedDict()
    simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def _adjoint(self):
        return Displace.create(-self.displacement, hs=self.space)

    _pseudo_inverse = _adjoint


@properties_for_args
class Squeeze(LocalOperator):
    r"""Unitary squeezing operator

    .. math::

        \Op{S}_{\rm hs}(\eta) =
        \exp {\left( \frac{\eta}{2} {\Op{a}_{\rm hs}^\dagger}^2 -
                     \frac{\eta^*}{2} {\Op{a}_{\rm hs}}^2 \right)}

    where :math:`\Op{a}_{\rm hs}` is the annihilation operator acting on the
    :class:`.LocalSpace` `hs`.

    Args:
        squeezing_factor (Scalar): the squeezing factor $\eta$
        hs (HilbertSpace or int or str): The Hilbert space on which the
            operator acts

    Printers should represent this operator with the default identifier::

        >>> Squeeze._identifier
        'Squeeze'

    A custom identifier may be define using `hs`'s `local_identifiers`
    argument.
    """
    _identifier = "Squeeze"
    _arg_names = ('squeezing_factor', )
    _rules = OrderedDict()
    simplifications = [implied_local_space(keys=['hs', ]), match_replace]

    def _adjoint(self):
        return Squeeze(-self.squeezing_factor, hs=self.space)

    _pseudo_inverse = _adjoint
