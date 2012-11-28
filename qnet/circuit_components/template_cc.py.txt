#!/usr/bin/env python
# encoding: utf-8
#This file was automatically created using QNET.

"""
{filename}

Created automatically by $QNET/bin/parse_qhdl.py
Get started by instantiating a circuit instance via:

    >>> {entity_name}()

"""

__all__ = ['{entity_name}']

from qnet.circuit_components.library import make_namespace_string
from qnet.circuit_components.component import Component
from qnet.algebra.circuit_algebra import cid, P_sigma, FB, SLH
import unittest
from sympy import symbols
{import_components}



class {entity_name}(Component):

    # total number of field channels
    CDIM = {CDIM}
    
    # parameters on which the model depends
    {param_attributes}
    _parameters = {param_names}

    # list of input port names
    PORTSIN = {PORTSIN}
    
    # list of output port names
    PORTSOUT = {PORTSOUT}

    # sub-components
    {sub_component_attributes}
    _sub_components = {sub_component_names}
    

    def _toSLH(self):
        return self.creduce().toSLH()
        
    def _creduce(self):

        {symbol_instantiation}

        return {symbolic_expression}

    @property
    def _space(self):
        return self.creduce().space


# Test the circuit
class Test{entity_name}(unittest.TestCase):
    """
    Automatically created unittest test case for {entity_name}.
    """

    def testCreation(self):
        a = {entity_name}()
        self.assertIsInstance(a, {entity_name})

    def testCReduce(self):
        a = {entity_name}().creduce()

    def testParameters(self):
        if len({entity_name}._parameters):
            pname = {entity_name}._parameters[0]
            obj = {entity_name}(name="TestName", namespace="TestNamespace", **{{pname: 5}})
            self.assertEqual(getattr(obj, pname), 5)
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

        else:
            obj = {entity_name}(name="TestName", namespace="TestNamespace")
            self.assertEqual(obj.name, "TestName")
            self.assertEqual(obj.namespace, "TestNamespace")

    def testToSLH(self):
        aslh = {entity_name}().toSLH()
        self.assertIsInstance(aslh, SLH)

if __name__ == "__main__":
    unittest.main()