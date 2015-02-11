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

We write 'creducible' instead of 'reducible' in order to distinguish the meaning from the definition of Gough and James,
who define reducible circuits as all circuits that can be decomposed into a concatenation of parts.
Creducibility is more general than reducibility since we allow for an expansion into any sort of non-trivial algebraic expression,
but in reverse, all reducible circuits are creducible.

Examples of creducible but primitive Components are:
KerrCavity, Relay, ...

non-creducible & primitive:
Beamsplitter, Phase, Displace

creducible & non-primitive:
Any parsed QHDL circuit

non-creducible & non-primitive:
None.

"""

from qnet.algebra.circuit_algebra import Circuit, Expression, tex, CannotConvertToABCD, substitute


#TODO update str insertion to str.format()




class Component(Circuit, Expression):
    """
    Base class for all circuit components, 
    both primitive components such as beamsplitters 
    and cavity models and also composite circuit models 
    that are built up from these.
    Via the creduce() method, an object can be decomposed into its parts.
    """

    CDIM = 0

    # Name of the component, only necessary if it carries 
    # its own physical degrees of freedom or if it is part of a circuit
    name = 'C'

    namespace = ''

    # ingoing port names
    PORTSIN = []

    # outgoing port names
    PORTSOUT = []

    _parameters = []
    _sub_components = []

    @property
    def _cdim(self):
        return self.CDIM

    def _all_symbols(self):
        return set(())

    def __init__(self, name = name, namespace = namespace, **kwparams):
        self.name = name
        self.namespace = namespace
        for pname, val in kwparams.items():
            if pname in self._parameters:
                setattr(self, pname, val)
            else:
                del kwparams[pname]
                print("Unknown parameter!")
        self._repr = "{}({!r},{!r}{})".format(self.__class__.__name__, self.name, self.namespace, "".join(", {}={!r}".format(k,v) for k,v in kwparams.items()))
        self._hash = hash((self.__class__, name, namespace, tuple(sorted(kwparams.items()))))


    def __repr__(self):
        return self._repr


    def __str__(self):
        if self.namespace:
            return "{}^{}".format(self.name, self.namespace)
        return self.name


    def _tex(self):
        """
        Return a tex representation of the component, including its parameters.
        """
        if self.namespace:
            return "{{{}}}^{{{}}}".format(tex(self.name), tex(self.namespace))
        return tex(self.name)

    def __hash__(self):
        return self._hash

    def _creduce(self):
        return self

    def _toABCD(self, linearize):
        return self.toSLH().toABCD(linearize)

    def _substitute(self, var_map):
        #TODO TEST
        all_names = self._parameters + self._sub_components
        all_namesubvals = [(n, substitute(getattr(self, n), var_map)) for n in all_names]
        return self.__class__(**dict(all_namesubvals))




        



class SubComponent(Circuit, Expression):
    """
    Class for the subcomponents of a reducible (but primitive) Component.
    """

    parent_component = None
    sub_index = 0

    def __init__(self, parent_component, sub_index):
        self.parent_component = parent_component
        self.sub_index = sub_index


    def __getattr__(self, attrname):
        try:
            return getattr(self.parent_component, attrname)
        except AttributeError:
            raise AttributeError(self.__class__.__name__ + "." + attrname)

    @property
    def PORTSIN(self):
        "Names of ingoing ports."
        offset = sum(self.parent_component.sub_blockstructure[:self.sub_index], 0)
        return self.parent_component.PORTSIN[offset: offset + self.cdim]

    @property
    def PORTSOUT(self):
        "Names of outgoing ports."
        offset = sum(self.parent_component.sub_blockstructure[:self.sub_index], 0)
        return self.parent_component.PORTSOUT[offset: offset + self.cdim]

    @property
    def _cdim(self):
        "Numbers of channels"
        return self.parent_component.sub_blockstructure[self.sub_index]

    def __str__(self):
        return str(self.parent_component) + "_" + str(self.sub_index)

    def __repr__(self):
        return "%s(%r, %d)" % (self.__class__.__name__, self.parent_component, self.sub_index)

    def _toSLH(self):
        raise NotImplementedError()

    def _tex(self):
        return "{%s}_{%d}" % (self.parent_component.tex(), self.sub_index)

    def __hash__(self):
        return hash((self.__class__, self.parent_component, self.sub_index))

    def _toABCD(self, linearize):
        raise CannotConvertToABCD()

    def _creduce(self):
        return self

    def _all_symbols(self):
        return self.parent_component.all_symbols()

    @property
    def _space(self):
        return self.parent_component.space

    def _substitute(self, var_map):
        raise NotImplementedError("Carry out substitution before calling creduce() or after converting to SLH")