#!/usr/bin/env python
# encoding: utf-8
"""
component.py

Created by Nikolas Tezak on 2012-01-03.
Copyright (c) 2012 . All rights reserved.

In the following, we distinguish between two properties of Components:
1) They may be 'reducible', i.e. they can be expressed as a circuit expression 
with irreducible operands.
2) They may be 'primitive', i.e. they cannot be specified via QHDL

Examples of reducible & primitive Components are:
KerrCavity, Relay, ...

irreducible & primitive:
Beamsplitter, Phase, Displace

reducible & non-primitive:
Any parsed QHDL circuit

irreducible & non-primitive:
None.


"""

from algebra.circuit_algebra import Circuit, Expression, tex

from collections import  OrderedDict
from algebra.abstract_algebra import mathematica

class Component(Circuit, Expression):
    """
    Base class for all circuit components, 
    both primitive components such as beamsplitters 
    and cavity models and also composite circuit models 
    that are built up from these.
    Via the reduce() method, an object can be decomposed into its parts.
    """

    CDIM = 0

    # Name of the component, only necessary if it carries 
    # its own physical degrees of freedom or if it is part of a circuit
    name = ''

    # parameters on which the model depends
    GENERIC_DEFAULT_VALUES = OrderedDict()

    # ingoing port names
    PORTSIN = []

    # outgoing port names
    PORTSOUT = []


    @property
    def sub_blockstructure(self):
        """
        If the component is reducible, this property should 
        be overwritten to give the correct block structure. 
        See some examples (kerr_cavity,...) for more details.
        """
        return self.CDIM,

    @property
    def cdim(self):
        return self.CDIM

    def __init__(self, name='Q', **params):
        self.name = name
        gparams = dict(self.GENERIC_DEFAULT_VALUES)
        gparams.update(params)

        for pname, val in gparams.items():
            setattr(self, pname, val)


    def toSLH(self):
        raise NotImplementedError(self.__class__.__name__)

    def __repr__(self):
        return "%s(%r%s)" % (self.__class__.__name__, self.name,
                             "".join(", %s = %r" % (k, getattr(self, k)) for k in sorted(self.GENERIC_DEFAULT_VALUES)))

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)


    def tex(self):

        """
        Return a tex representation of the component, including its parameters.
        """
        try:
            tex_string = r"{%s}\left({%s}%s\right)" % (self.tex_name,
                                                       tex(self.name),
                                                       "".join(", %s" % tex(getattr(self, k))
                                                       for k in sorted(self.GENERIC_DEFAULT_VALUES)))
            return tex_string
        except Exception:
            return tex(self.name)
        
    def mathematica(self):
        return  "%s[%s, %s]" % (self.__class__.__name__, mathematica(self.name), 
                                ", ".join(["Rule[%s,%s]" % (str(gp), mathematica(getattr(self, str(gp)))) for gp in self.GENERIC_DEFAULT_VALUES]))

        return tex(self.name)
        # raise NotImplementedError(self.__class__.__name__) 



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
    def cdim(self):
        "Numbers of channels"
        return self.parent_component.sub_blockstructure[self.sub_index]

    def __str__(self):
        return str(self.parent_component) + "_" + str(self.sub_index)

    def __repr__(self):
        return "%s(%r, %d)" % (self.__class__.__name__, self.parent_component, self.sub_index)

    def toSLH(self):
        raise NotImplementedError()

    def tex(self):
        return "{%s}_{%d}" % (self.parent_component.tex(), self.sub_index)

    def mathematica(self):
        return "SubBlock[%s, %d]" % (mathematica(self.parent_component, self.sub_index))
