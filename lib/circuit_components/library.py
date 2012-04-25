#!/usr/bin/env python
# encoding: utf-8
import os
my_dir = os.path.dirname(__file__)

ARCH_PREFIX = 'arch_'

def make_namespace_string(name_space, sub_name):
    if name_space == '':
        return sub_name
    return sub_name  + "__" + name_space
        

# def retrieve_component(entity_identifier, cdim, name_space, architecture = 'default', **generic_params):
#     try:
#         component_module = __import__("circuit_components.%s_circuit" % entity_identifier, fromlist = ['circuit_components'])
#     except ImportError, e:
#         import sys
#         print sys.path
#         raise ImportError("Could not retrieve Circuit file: %s" % e)
#         
#     if component_module.CDIM != cdim:
#         raise Exception('Specified number of in/out channels does not match specification in imported module.')
#     
#     architectures = {}
#     for k, v in component_module.__dict__.items():
#         if k.startswith(ARCH_PREFIX) and callable(v):
#             architectures[k[len(ARCH_PREFIX):]] = v
#             
#     if not architecture in architectures:
#         raise Exception('Architecture named %s not found for %s.\nPlease define a function named %s%s.'\
#                         % (architecture, entity_identifier, ARCH_PREFIX, architecture))
#     
#     symbolic, var_map = architectures[architecture](name_space = name_space, **generic_params)
#     
#     return symbolic, var_map
        

# def invert_dict(d):
#     return dict(((v,k) for k,v in d.items()))


def _make_default_value_string(name, tp, default):
    if default is not None:
        return str(default)
    if tp == 'real':
        return 'symbols(%r, real = True, each_char = False)' % (name,)
    return 'symbols(%r, each_char = False)' % (name,)

import re
camelcase_to_underscore = lambda st: re.sub('(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))', '_\\1', st).lower().strip('_')
    
def getCDIM(component_name):
    try:
        component_module = __import__("circuit_components.%s_cc" % camelcase_to_underscore(component_name), fromlist = ['circuit_components'])
    except ImportError, e:
        raise ImportError("Could not retrieve Circuit file: %s" % e)
    return getattr(component_module,component_name).CDIM


def write_component(entity, architectures, default_architecture = None):
    
    arch_template = """
    def arch_{arch}(self):
        # import referenced components
        {import_components}
        
        # instantiate circuit components
        {symbol_assignments}
        
        return {symbolic_expression}
    """
    
    architecture_strings = []
    for arch_name, arch in architectures.items():
        arch_circuit, circuit_symbols, instance_assignments = arch.to_circuit()
        
        import_component_strings = []
        for component_name in arch.components:
            import_component_strings.append("from circuit_components.%s_cc import %s" % (camelcase_to_underscore(component_name), component_name))
        import_components = "\n        ".join(import_component_strings)

        symbol_assignment_strings = []
        for instance_name in circuit_symbols:
            component, generic_map = instance_assignments[instance_name][0 : 2]
            generic_assignments = "".join([", {comp_generic} = self.{entity_generic}".format(comp_generic = cg, entity_generic = eg) 
                                                                            for cg, eg in generic_map.items()])
                                                                            
            symbol_assignment_strings.append("{sname} = {cname}(make_namespace_string(self.name, '{sname}'){generic_assignments})".format(
                                                sname = instance_name, 
                                                cname = component.identifier, 
                                                generic_assignments = generic_assignments))
        symbol_assignments = "\n        ".join(symbol_assignment_strings)        
        
        architecture_strings.append(arch_template.format(arch = arch_name, 
                                                        import_components = import_components, 
                                                        symbol_assignments = symbol_assignments, 
                                                        symbolic_expression = str(arch_circuit)))

    
    
    with open(my_dir + '/template_cc.py.txt', 'r') as template:
        file_template = template.read()
    
    cc_file_path = my_dir + "/" + camelcase_to_underscore(entity.identifier) + "_cc.py"
    
    # backup existing files
    if os.path.exists(cc_file_path):
        j = 0
        while(os.path.exists(cc_file_path + ".backup_%d" % j)):
            j += 1
        os.rename(cc_file_path, cc_file_path + ".backup_%d" % j)
    
    
    template_params = {
        "filename" : os.path.basename(cc_file_path),
        "entity_name": entity.identifier,
        "CDIM" : entity.cdim,
        "GENERIC_DEFAULT_VALUES": ",\n        ".join(("%s = %s" % (generic_name, _make_default_value_string(generic_name, gtype, default_value)) for (generic_name, (gtype, default_value)) in entity.generics.items())),
        "PORTSIN" : entity.in_port_identifiers,
        "PORTSOUT" : entity.out_port_identifiers,
        "architectures" : "\n    ".join(architecture_strings),
        "first_arch_name": architectures.keys().pop(),
    }
#     
#     """#!/usr/bin/env python
# # encoding: utf-8
# # This file was initially generated from a QHDL source.
# 
# from circuit_components.library import retrieve_component, make_namespace_string
# from algebra.circuit_algebra import CSymbol, P_sigma, cid, FB
# from sympy.core.symbol import symbols
# 
# CDIM = %d
# GENERIC_DEFAULT_VALUES = {%s}
# 
# """ % (entity.cdim, )
# 
# 
#     #for every architecture write architecture method
#     for arch_name, arch in architectures.items():
#         assert arch.entity == entity
#         arch_circuit, circuit_symbols, instance_assignments = arch.to_circuit()
#         
#         src_file.write("def %s%s(name_space = '', **generic_params):\n" % (ARCH_PREFIX, arch_name))    
#         src_file.write("    var_map = {}\n")
#         if len(entity.generics) > 0:
#             src_file.write("\n    # generic parameters\n")
#         for generic_name in entity.generics:
#             src_file.write("    %s = generic_params.get('%s', GENERIC_DEFAULT_VALUES['%s'])\n" % (generic_name, generic_name, generic_name))
#         
#         src_file.write("\n    # instances\n")
#         for instance_name in circuit_symbols:
#             component, generic_map = instance_assignments[instance_name][0 : 2]
# 
#             src_file.write("    %s, %s_var_map = retrieve_component('%s', %d, make_namespace_string(name_space,'%s'), %s)\n" \
#                         % (instance_name, instance_name, component.identifier, \
#                             component.cdim, instance_name, \
#                             ", ".join(("%s = %s" % (c_g, e_g)  for c_g, e_g in generic_map.items())) ))
#             src_file.write("    var_map.update(%s_var_map)\n\n" % (instance_name,))
#             
#         src_file.write("    %s_symbolic = %s\n" % (arch_name, arch_circuit))
#         src_file.write("    return %s_symbolic, var_map\n\n\n" % arch_name)
#         # src_file.write("# %s" % (arch_circuit))  
#     if default_architecture != None:
#         src_file.write("%sdefault = %s%s\n\n" % (ARCH_PREFIX, ARCH_PREFIX, default_architecture)) 
#     else:
#         src_file.write("%sdefault = %s%s\n\n" % (ARCH_PREFIX, ARCH_PREFIX, architectures.keys().pop())) 
#     src_file.write("def test_evalf():\n")
#     src_file.write("    a, va = %sdefault()\n" % ARCH_PREFIX)
#     src_file.write("    print a\n")
#     src_file.write("    sa = a.substitute(va)\n")
#     src_file.write('    print "-"*30\n')
#     src_file.write("    print sa\n")
#     src_file.write('    print "-"*30\n')        
#     src_file.write("    print sa.evalf()\n\n")
#     
#     src_file.write("if __name__ == '__main__':\n")
#     src_file.write("    test_evalf()\n"
#     
    
    
    with open(cc_file_path, "w") as src_file:
        src_file.write(file_template.format(**template_params))
        
    return cc_file_path
    