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
This module features some helper functions for automatically creating and managing a library of circuit component definition files.
"""
import os
import re

MODULE_DIR = os.path.dirname(__file__)

def make_namespace_string(namespace, sub_name):
    """
    Make a namespace string by combining a namespace string with a new name.

    :param namespace: The namespace so far
    :type namespace: str
    :param sub_name: The additional name to add/
    :type sub_name: str
    :return: The combined namespace
    :rtype: str
    """
    if namespace == '':
        return sub_name
    return namespace + "." + sub_name


camelcase_to_underscore = lambda st: re.sub('(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))', '_\\1', st).lower().strip('_')
"""Convert a camelcase entity name into an appropriate underscore name to import its corresponding module"""


def getCDIM(component_name):
    """
    Get the channel dimension of a referenced subcomponent

    :param component_name: The entity name of the component
    :type component_name: str
    :return: The channel dimension of the component.
    :rtype: int
    """
    try:
        component_module = __import__("qnet.circuit_components.%s_cc" % camelcase_to_underscore(component_name),
            fromlist=['qnet.circuit_components'])
    except ImportError as e:
        raise ImportError("Could not retrieve Circuit file: %s" % e)
    return getattr(component_module, component_name).CDIM

def _make_default_value_string(name, tp, default):
    if default is not None:
        return str(default)
    if tp == 'real':
        return 'symbols(%r, real = True)' % (name,)
    return 'symbols(%r)' % (name,)

def write_component(entity, architectures, local = False):
    """
    Write a new entity definition to a python module file.

    :param entity: The entity object
    :type entity: :py:class:`qnet.qhdl.qhdl.Entity`
    :param architectures: A dictionary of architectures ``dict(name = architecture)`` associated with the entity.
    :type architectures: dict
    :param local: Whether or not to store the created module in the current/local directory or install it in :py:module:``qnet.circuit_components``, default = ``False``
    :type local: bool
    :return: The filename of the new module.
    :rtype: str
    """

    if len(architectures) > 1:
        print("Warning: using only first architecture")
    arch = architectures.values().pop(0)

    arch_circuit, circuit_symbols, instance_assignments = arch.to_circuit()

    import_component_strings = []
    for component_name in arch.components:
        import_component_strings.append("from qnet.circuit_components.%s_cc import %s" % (camelcase_to_underscore(component_name), component_name))
    import_components = "\n".join(import_component_strings)

    symbol_assignment_strings = []
    for instance_name in sorted(circuit_symbols):
        component, generic_map = instance_assignments[instance_name][0 : 2]

        generic_assignments = "".join([", {comp_generic} = {entity_generic}".format(comp_generic = cg, entity_generic = "self." + eg if isinstance(eg, str) else str(eg))
                                                                        for cg, eg in generic_map.items()])

        symbol_assignment_strings.append("""
    @property
    def {sname}(self):
        return {cname}(make_namespace_string(self.name, '{sname}'){generic_assignments})
""".format(
            sname = instance_name,
            cname = component.identifier,
            generic_assignments = generic_assignments))

    symbol_assignments = "".join(symbol_assignment_strings)
    symbol_instantiation = ", ".join(sorted(circuit_symbols.keys())) + " = " + ", ".join("self."+k for k in sorted(circuit_symbols.keys()))
    symbolic_expression = str(arch_circuit)
    
    with open(MODULE_DIR + '/_template_cc.py', 'r') as template:
        file_template = template.read()


    if local:

        cc_file_path = camelcase_to_underscore(entity.identifier) + "_cc.py"
    else:

        cc_file_path = MODULE_DIR + "/" + camelcase_to_underscore(entity.identifier) + "_cc.py"
    
    # backup existing files
    # TODO add code to rescue doc-strings
    if os.path.exists(cc_file_path):
        j = 0
        while(os.path.exists(cc_file_path + ".backup_%d" % j)):
            j += 1
        os.rename(cc_file_path, cc_file_path + ".backup_%d" % j)
    
    
    template_params = {
        "filename" : os.path.basename(cc_file_path),
        "entity_name": entity.identifier,
        "CDIM" : entity.cdim,
        "param_attributes": "\n    ".join(("%s = %s" % (generic_name, _make_default_value_string(generic_name, gtype, default_value)) for (generic_name, (gtype, default_value)) in entity.generics.items())),
        "param_names": repr(sorted(entity.generics.keys())),
        "PORTSIN" : entity.in_port_identifiers,
        "PORTSOUT" : entity.out_port_identifiers,
        "sub_component_attributes" : symbol_assignments,
        "sub_component_names" : repr(sorted(circuit_symbols.keys())),
        "symbol_instantiation": symbol_instantiation,
        "symbolic_expression": symbolic_expression,
        "import_components":import_components,
    }
    
    with open(cc_file_path, "w") as src_file:
        src_file.write(file_template.format(**template_params))
        
    return cc_file_path
    
