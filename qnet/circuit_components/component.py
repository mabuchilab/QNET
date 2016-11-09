#This file is part of QNET.
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
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################

"""
We distinguish between two independent properties of Components:

1) They may be 'creducible', i.e. they can be expressed as a circuit expression
of sub components.

2) They may be 'primitive', i.e. they cannot be specified via QHDL

We write 'creducible' instead of 'reducible' in order to distinguish the
meaning from the definition of Gough and James, who define reducible circuits
as all circuits that can be decomposed into a concatenation of parts.
Creducibility is more general than reducibility since we allow for an expansion
into any sort of non-trivial algebraic expression, but in reverse, all
reducible circuits are creducible.

Examples of creducible but primitive Components are:
KerrCavity, Relay, ...

non-creducible & primitive:
Beamsplitter, Phase, Displace

creducible & non-primitive:
Any parsed QHDL circuit

non-creducible & non-primitive:
None.

"""
from abc import ABCMeta, abstractmethod
from collections import OrderedDict


from qnet.algebra.hilbert_space_algebra import TrivialSpace
from qnet.algebra.circuit_algebra import Circuit, CannotConvertToABCD
from qnet.algebra.abstract_algebra import Expression, substitute


class Component(Circuit, Expression, metaclass=ABCMeta):
    """Base class for all circuit components,
    both primitive components such as beamsplitters
    and cavity models and also composite circuit models
    that are built up from these.
    Via the creduce() method, an object can be decomposed into its parts.
    """

    CDIM = 0

    # ingoing port names
    PORTSIN = []

    # outgoing port names
    PORTSOUT = []

    _parameters = []
    _sub_components = []

    @property
    def cdim(self):
        return self.CDIM

    def _all_symbols(self):
        return set(())

    def __init__(self, name, **kwargs):
        self.name = name
        for pname, val in kwargs.items():
            if pname in self._parameters:
                setattr(self, pname, val)
            else:
                del kwargs[pname]
                print("Unknown parameter!")
        super().__init__(name, **kwargs)

    @property
    def args(self):
        return (self.name, )

    @property
    def kwargs(self):
        res = OrderedDict()
        for key in self._parameters:
            try:
                res[key] = self.key
            except AttributeError:
                pass
        return res

    @abstractmethod
    def _toSLH(self):
        raise NotImplementedError()

    @property
    def space(self):
        return TrivialSpace

    def _creduce(self):
        return self

    def _toABCD(self, linearize):
        return self.toSLH().toABCD(linearize)

    def _substitute(self, var_map):
        all_names = self._parameters + self._sub_components
        all_namesubvals = [(n, substitute(getattr(self, n), var_map))
                           for n in all_names]
        return self.__class__(**dict(all_namesubvals))


class SubComponent(Circuit, Expression, metaclass=ABCMeta):
    """Class for the subcomponents of a reducible (but primitive) Component."""

    parent_component = None
    sub_index = 0

    def __init__(self, parent_component, sub_index):
        self.parent_component = parent_component
        self.sub_index = sub_index
        super().__init__(parent_component, sub_index)

    def __getattr__(self, attrname):
        try:
            return getattr(self.parent_component, attrname)
        except AttributeError:
            raise AttributeError(self.__class__.__name__ + "." + attrname)

    @property
    def PORTSIN(self):
        "Names of ingoing ports."
        offset = sum(self.parent_component.sub_blockstructure[:self.sub_index],
                     0)
        return self.parent_component.PORTSIN[offset: offset + self.cdim]

    @property
    def PORTSOUT(self):
        "Names of outgoing ports."
        offset = sum(self.parent_component.sub_blockstructure[:self.sub_index],
                     0)
        return self.parent_component.PORTSOUT[offset: offset + self.cdim]

    @property
    def cdim(self):
        "Numbers of channels"
        return self.parent_component.sub_blockstructure[self.sub_index]

    def __str__(self):
        return str(self.parent_component) + "_" + str(self.sub_index)

    def __repr__(self):
        return "%s(%r, %d)" % (self.__class__.__name__, self.parent_component,
                               self.sub_index)

    @abstractmethod
    def _toSLH(self):
        raise NotImplementedError()

    @property
    def args(self):
        return self.parent_component, self.sub_index

    def tex(self):
        return "{%s}_{%d}" % (self.parent_component.tex(), self.sub_index)

    def _toABCD(self, linearize):
        raise CannotConvertToABCD()

    def _creduce(self):
        return self

    def all_symbols(self):
        return self.parent_component.all_symbols()

    @property
    def space(self):
        return self.parent_component.space

    def _substitute(self, var_map):
        raise NotImplementedError("Carry out substitution before calling "
                                  "creduce() or after converting to SLH")
