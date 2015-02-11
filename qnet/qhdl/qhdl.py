# coding=utf-8
#This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
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
This module contains the code to convert a circuit specified in QHDL into a Gough-James circuit expression.

The other module in this package :py:mod:`qhdl_parser` implements an actual parser for the qhdl source text, while this file then
converts structured netlist information into a circuit expression.

For more details on the QHDL syntax, see :ref:`qhdl_syntax`.
"""

from collections import OrderedDict
from qnet.circuit_components.library import getCDIM




def my_debug(msg):
    pass
    # print(msg)


class QHDLError(Exception):
    pass

class QHDLObject(object):
    
    def to_python(self):
        return self.__repr__()
    
    def to_qhdl(self):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError


PORT_DIRECTIONS = ('in', 'out')
SIGNAL_TYPES = ('fieldmode')
GENERIC_TYPES = ('int', 'real', 'complex')

def gtype_compatible(c_t, g_t):
    return GENERIC_TYPES.index(c_t) >= GENERIC_TYPES.index(g_t)

class BasicInterface(QHDLObject):
    
    
    def __repr__(self):
        return "%s('%s', %r, %r)" % (self.__class__.__name__, self.identifier, self.generics, self.ports)
    
    def to_qhdl(self,tab_level):
        pass
    
    def generics_to_qhdl(self, tab_level):
        if len(self.generics) == 0:
            return ""
        default_formatter_by_type = {float: (lambda v: '%g' % v),
                                    int: (lambda v: '%d' % v),
                                    long: (lambda v: '%d' % v),
                                    complex:(lambda v: "(%g, %g)" % (v.real, v.imag))}
        default_str = lambda default: " := %s" % default_formatter_by_type[type(default)](default) \
                                            if default else ""
        g_str = "generic (%s);\n" % (";\n    ".join(["%s: %s%s" % (identifier, gtype, default_str(default)) \
                                                for identifier,(gtype, default) in self.generics.items()]))
        return (tab_level*"\t") + g_str
    
    def ports_to_qhdl(self, tab_level):
        
        in_str = "%s: in fieldmode" % ",\n    ".join(self.port_identifiers[ : self.cdim/2])
        out_str = "%s: out fieldmode" % ",\n    ".join(self.port_identifiers[self.cdim/2 : ])
        p_str = "port (%s;\n    %s);" % (in_str, out_str)
        
        return (tab_level*"\t") + p_str
    
#    @property
#    def in_port_identifiers(self):
#        return self.in_ports.keys()
#        
#    @property
#    def out_port_identifiers(self):
#        return self.out_ports.keys()
#    
    cid = 0
    in_port_identifiers = []
    out_port_identifiers = []
    inout_port_identifiers = []
    
    
    def __init__(self, identifier, generics, ports):
        self.identifier = identifier
        self.cid = 0
        self.generics = OrderedDict()
        self.in_port_identifiers = []
        self.out_port_identifiers = []
        self.ports = OrderedDict()
        self.inout_port_identifiers = []
        
        for (identifier_list, gtype, default_val) in generics:
            if gtype not in GENERIC_TYPES:
                raise QHDLError("Generic type of generics %s is %s, but must be one of the following: %r" \
                                         % (", ".join(identifier_list), gtype, GENERIC_TYPES))
            for gid in identifier_list:
                if gid in self.generics:
                    raise QHDLError("Generic identifier non-unique: %s" % identifier)
                self.generics[gid] = gtype, default_val
        
        
#        # rewrite the port list and replace inout ports by a separate in and out port 
#        for (identifier_list, direction, signal_type) in list(ports):
#            if direction == "inout":
#                self.inout_port_names += identifier_list
#                identifier_list_i = [identifier + "_i" for identifier in identifier_list]
#                identifier_list_o = [identifier + "_o" for identifier in identifier_list]
#                ports = [(identifier_list_i, 'in', signal_type), (identifier_list_o, 'out', signal_type)]
#        
        
        for (identifier_list, direction, signal_type) in ports:
            if direction not in ('in', 'out','inout'):
                raise QHDLError(str((identifier_list, direction, signal_type)))
            
            if signal_type != "fieldmode":
                raise QHDLError("We currently only support signals of type 'fieldmode': " + str(identifier_list))
            for pid in identifier_list:
                if pid in self.generics:
                    raise QHDLError("There already exists a generic with the same identifier : %s" % pid)
                if pid in self.ports:
                    raise QHDLError("Port identifier non-unique: %s" % pid)
                
                if direction == 'inout':
                    self.inout_port_identifiers.append(pid)
                elif direction == 'in':
                    self.in_port_identifiers.append(pid)
                else: # direction == 'out'
                    self.out_port_identifiers.append(pid)
                
                self.ports[pid] = direction, signal_type
                    
                
                
#        if not len(self.in_ports) == len(self.out_ports):
#            raise QHDLError('The numbers of input and output channels do not match.')
#        self._port_identifiers = tuple(port_identifiers)
#        self._generic_identifiers = tuple(generic_identifiers)
                                    
    
#    @property
#    def cdim(self):
#        return len(self.in_ports)
#
    @property
    def port_identifiers(self):
        """The port_identifiers property."""
        return self.inout_port_identifiers + self.in_port_identifiers + self.out_port_identifiers
#
#    pids = port_identifiers

    @property
    def generic_identifiers(self):
        """The generic_identifiers property."""
        return self.generics.keys()

    gids = generic_identifiers
        
    
#    def __getitem__(self, key):
#        if key in self.ports:
#            return self.ports[key]
#        elif key in self.generics:
#            return self.generics[key]
#        else:
#            raise KeyError(str(key))
#    
#    def get(self, key, default_val = None):
#        try:
#            return self[key]
#        except KeyError:
#            return default_val
#    
#    def has_generic(self, gid):
#        return gid in self.generics
#    
#    def has_port(self, pid):
#        return pid in self.ports
        
        

class Entity(BasicInterface):
    def to_qhdl(self, tab_level = 0):
        ret_str =  """entity %s is
    %s%s
end entity %s;""" % (self.identifier, self.generics_to_qhdl(tab_level), self.ports_to_qhdl(tab_level), self.identifier)
        return (tab_level*"\t") + ret_str


class Component(BasicInterface):
    
    def to_qhdl(self, tab_level = 0):
        ret_str =  """component %s
    %s%s
end component %s;""" % (self.identifier, self.generics_to_qhdl(tab_level), self.ports_to_qhdl(tab_level), self.identifier)
        return (tab_level*"\t") + ret_str



def dict_keys_sorted_by_val(dd):
    return sorted(dd.keys(), key = dd.get)

class Architecture(QHDLObject):
    global_inout = {}
    global_out = {}
    global_in = {}
    inout_to_signal = {} 
    out_to_signal = {}
    in_to_signal = {}
    signal_to_global_in = {}
    signal_to_global_out = {}
    
    signals = []
    lossy_signals = []
    
    
    
    def __init__(self, identifier, entity, components, signals, assignments, global_assignments = {}):
        self.identifier = identifier
        self.entity = entity
        self.components = OrderedDict()
        self.signals = []
        self.lossy_signals = []
        self.instance_assignments = OrderedDict()
        self._circuit_data = False
        
        # lookuptables format
        # (instance_name, instance_port_name)
        #       => ((target_instance_name|'entity'), port_name, port_id, (connecting_signal_name|None))
#        mediated_inport_map = {}
#        mediated_outport_map = {}
        self.global_inout = {}
        self.global_out = {}
        self.global_in = {}
        self.inout_to_signal = {}
        self.out_to_signal = {}
        self.in_to_signal = {}
        self.signal_to_global_in = {}
        self.signal_to_global_out = {}
        
        
        #process components
        for component in components:
            #check for duplicate component identifiers
            if component.identifier in self.components:
                raise QHDLError('Component ID non-unique: %s' % component.identifier)
            component.cdim = getCDIM(component.identifier)
            self.components[component.identifier] = component
            
        
        #process signals
        for signal_ids, signal_type in signals:
            if signal_type not in ('fieldmode','lossy_fieldmode'):
                raise QHDLError("Currently only fieldmode and lossy_fieldmode are accepted as signal types: \n %s : %s" % (", ".join(signal_ids), signal_type))
            
            for sid in signal_ids:
                #check if signal identifier coincides with entity port name
                if sid in entity.port_identifiers:
                    raise QHDLError('Signal identifier already used as an entity port identifier: %s' % sid)
                
                #check for duplicate signal identifiers
                if (sid in self.signals) or (sid in self.lossy_signals):
                    raise QHDLError('Signal identifier non-unique: %s' % sid)
                #every signal can only connect two ports of component instances,
                #one in-port and one out-port
                if signal_type == 'fieldmode':
                    self.signals.append(sid)
                else:
                    self.lossy_signals.append(sid)
                    
        #process instance definitions, assignments
        for instance_name, component_name, generic_map, port_map in assignments:
            #check for duplicate instance identifier
            if instance_name in self.instance_assignments:
                raise QHDLError('Instance name non-unique: %s' % instance_name)
            
            #check if referenced component (interface) exists
            component = self.components.get(component_name, False)
            if not component:
                raise QHDLError('Bad component identifier: %s' % component_name)
            
            #convert generic map (a,b,c,...) into explicit generic map (q=>a, r=>b,...)
            #based on the generics of the component
            if not isinstance(generic_map, dict):
                #number of assignments must match number of generics in component
                if not len(generic_map) == len(component.generic_identifiers):
                    raise QHDLError('All generics of component %s must be assigned' % component.identifier)
                generic_map = dict(zip(component.generic_identifiers, generic_map))
                
            # if no generic map statement but the component definition has some generics defined
            elif generic_map == {} and len(component.generics) > 0:
                # assert that there are default values for all generics
                for gname, (_, default) in component.generics.items():
                    if default is None:
                        raise QHDLError(('No default value defined for generic %s of component %s,\n' % (gname, component.identifier)) 
                                    + 'either define this or add a generic map statement to the instance assigment of %s' % (instance_name,))
                
            
            #if generic map is already in explicit form, assert that the
            #assigned q,r,... coincide those defined in the component
            elif sorted(generic_map.keys()) != sorted(component.generics.keys()):
                raise QHDLError('All generics of component %s must be assigned' % component.identifier)
            
            #make sure referenced a,b,c,... exist as generics in the entity
            for name_in_c, name_in_e in generic_map.items():
                is_number = isinstance(name_in_e, (int, float, complex))
                if not is_number:
                    entity_g = entity.generics.get(name_in_e, False)
                    if not entity_g:
                        raise QHDLError('Entity %s does not define a generic of name %s and it cannot be parsed to a numeric value.' % (entity.identifier, name_in_e))
                
                    component_g = component.generics.get(name_in_c, False)
                
                    #probably redundant
                    if not component_g:
                        raise QHDLError('Component %s does not define a generic of name %s' % (component.identifier, name_in_c))
                        #check that generics have compatible type
                        #e.g. component_generic_type == real, entity_generic_type == int is okay
                        #because an int may be cast to a real, however the other way around does not always work!
                    if not gtype_compatible(component_g[0], entity_g[0]) :
                        raise QHDLError('Mapped generics are of incompatible type.' \
                                        + 'Change the generic\'s type in the entity to %s' % component_g[0])
            
            
            # TODO rewrite port map based on whether a port is inout
            
            #convert port map (a,b,c,...) into port map (q=>a, r=>b,...)
            #based on the ports of the component
            if not isinstance(port_map, dict):
                if len(port_map) != len(component.port_identifiers):
                    raise QHDLError('All ports of instance %s(%s) must be mapped' % (instance_name, component.identifier))
                port_map = dict(zip(component.port_identifiers, port_map))
            
            #if port map is already in explicit form, assert that the
            #assigned q,r,... coincide those defined in the component
#            elif sorted(port_map.keys()) != sorted(component.ports.keys()):
#                raise QHDLError('All ports of instance %s(%s) must be mapped' % (instance_name, component.identifier))
            

            
            #create lookup tables
            for name_in_c, name_in_e in port_map.items():
                
                
                #any referenced a,b,c,... must either exist
                #as a signal in the architecture or a port of the entity
                entity_p = entity.ports.get(name_in_e, False)
                try:
                    self.signals.index(name_in_e)
                    signal = name_in_e
                    
                except ValueError:
                    try:
                        self.lossy_signals.index(name_in_e)
                        signal = name_in_e
                    except ValueError:
                        signal = False
                
                if not entity_p and not signal:
                    if name_in_e == 'OPEN':
                        continue
                    else:
                        print(self.signals, self.lossy_signals)
                        
                        raise QHDLError('The entity %s does not define a port\
and the architecture %s \
does not any define any signal of \
name %s ' % (entity.identifier, self.identifier, name_in_e))
                if signal and entity_p:
                    raise QHDLError("Duplicate name for a signal and an entity port: %s" % name_in_e)                    

                
                component_p = component.ports.get(name_in_c, False)
                #probably redundant
                if not component_p:
                    raise QHDLError('Component %s does not define a port of name %s' % (component.identifier, name_in_c))
                
                c_dir, _ = component_p
                
                if entity_p:
                    e_dir, _ = entity_p
                    if c_dir ==  "inout":
                        raise QHDLError('Component inout port %s.%s must be connected to signal' % (component.identifier, name_in_c))
                    
                    if not e_dir == c_dir:
                        raise QHDLError('Component port %s.%s must be connected to entity port of same direction' % (component.identifier, name_in_c))
                    
#                    if c_dir == "inout":
#                        self.global_inout[(instance_name, name_in_c)] = name_in_e
#                    
                    if c_dir == "in":
                        self.global_in[(instance_name, name_in_c)] = name_in_e
                    else:
                        assert c_dir == "out"
                        self.global_out[(instance_name, name_in_c)] = name_in_e
                else:
                    if c_dir == "inout":
                        self.inout_to_signal[(instance_name, name_in_c)] = name_in_e
                    elif c_dir == "in":
                        self.in_to_signal[(instance_name, name_in_c)] = name_in_e
                    else:
                        assert c_dir == "out"
                        self.out_to_signal[(instance_name, name_in_c)] = name_in_e
                self.instance_assignments[instance_name] = component, generic_map, port_map
        
        for source_id, target_id in global_assignments.items():
            # TODO: handle signals to INOUT ports
            if target_id in self.entity.out_port_identifiers:
                if not source_id in self.signals:
                    raise QHDLError('Global Out-Ports may only be assigned to signals')
                self.signal_to_global_out[source_id] = target_id
                
            elif target_id in self.signals:
                if not source_id in self.entity.in_port_identifiers:
                    raise QHDLError('Signals can only be assigned to global In-Ports')
                self.signal_to_global_in[target_id] = source_id
            else:
                raise QHDLError('Global Assignment Error: %s => %s' % (source_id, target_id))


    def to_circuit(self, identifier_postfix = ''):
        """
        Compute a circuit algebra expression from the QHDL code and return the
        circuit expression, the all_symbols appearing in it and the component instance assignments
        """
        
        if self._circuit_data:
            return self._circuit_data
        from qnet.algebra import circuit_algebra as ca


        # initialize trivial circuit
        circuit = ca.cid(0)

        II = []
        OO = []

        if len(self.lossy_signals):
            if not self.components.get('Beamsplitter', False):
                self.components['Beamsplitter'] = Component('Beamsplitter', [('theta','real')],[(['In1','In2'],'in','fieldmode'),(['Out1','Out2'],'out','fieldmode')])
                self.components['Beamsplitter'].cdim = 2
        
        
        OPEN = object()
        #create all_symbols for all instances
        circuit_symbols = {}
        for (instance_name, (component, _, _)) in self.instance_assignments.items():
            QQ = ca.CircuitSymbol(instance_name + identifier_postfix, component.cdim)
            circuit_symbols[instance_name] = QQ
            
            assert component.inout_port_identifiers == []
            
            circuit  = circuit + QQ
#            II = II + [(instance_name, port_name + "_i" ) for port_name in component.inout_port_identifiers]
            II = II + [(instance_name, port_name) for port_name in component.in_port_identifiers]
            
#            OO = OO + [(instance_name, port_name + "_o") for port_name in component.inout_port_identifiers]
            OO = OO + [(instance_name, port_name) for port_name in component.out_port_identifiers]
            
            if len(component.in_port_identifiers) + len(component.inout_port_identifiers) < component.cdim:
                II = II + [(OPEN,OPEN)] * ( component.cdim - len(component.in_port_identifiers) - len(component.inout_port_identifiers))
            
            if len(component.out_port_identifiers) < component.cdim:
                OO = OO + [(OPEN,OPEN)] * ( component.cdim - len(component.out_port_identifiers) - len(component.inout_port_identifiers))
        
        
        # Add loss-beamsplitters
        for k, s in enumerate(self.lossy_signals):
        
#            while "LSS%s_%d%s" % (s, j, identifier_postfix) in circuit_symbols:
#                j += 1
            LBS = ca.CircuitSymbol("LSS_%s%s" % (s,identifier_postfix), 2)
            circuit = circuit + LBS
            
            
            II = II + [('LSS_%s' % s, 'In1'),(OPEN, OPEN)]
            OO = OO + [('LSS_%s' % s, 'Out1'),(OPEN, OPEN)]
            
            self.signals.append(s+"__from_loss")
            self.signals.append(s)
            
            
            # modify assignment of original component that leads into signal
            try:
                # exploit enforced order of dictionaries
                jj = list(self.in_to_signal.values()).index(s)
                ipnames = list(self.in_to_signal.keys())[jj]
                assert self.in_to_signal[ipnames] == s
                self.in_to_signal[ipnames] = s + "__from_loss"
            except ValueError:
                jj = list(self.global_out.values()).index(s)
                ipnames = list(self.global_out.keys())[jj]
                assert self.global_out[ipnames] == s
                self.global_out[ipnames] = s + "__from_loss"
            # Update lookup tables
            self.in_to_signal[('LSS_%s' % s, 'In1')] = s
            self.out_to_signal[('LSS_%s' % s, 'Out1')] = s + "__from_loss"
            
            # Create artificial instance assignment
            self.instance_assignments['LSS_%s' % s] = self.components['Beamsplitter'], {'theta': 'theta_LS%d' % k},{"In1": s, "Out1": s + "__from_loss"}
            circuit_symbols['LSS_%s' % s] = LBS
            self.entity.generics['theta_LS%d' % k] = "real", None
            
        
        assert circuit.cdim == len(OO) == len(II)
        SS = list(self.signals)
        
        
        # Add signals as passthru lines below rest
        circuit = circuit + ca.cid(len(SS))
#        print(circuit)
        
        # Do feedback from instance output to signals
        SSp = list(SS)
        OOp = list(OO)
        M = len(OO)
        for iname, pname in OO:
            if iname is OPEN:
                continue
            sname = self.out_to_signal.get((iname, pname), False)
            if sname:
                k = OOp.index((iname, pname))
                l = SSp.index(sname) + M
                
                circuit = circuit.feedback(k,l)
                SSp.remove(sname)
                OOp.remove((iname, pname))
#        print(circuit)
        # Do feedback from signal output to instance inputs
        IIp = list(II)
        SSpp = list(SS)
        Mf = len(OOp)
        
        for iname, pname in II:
            if iname is OPEN:
                continue
            sname = self.in_to_signal.get((iname, pname), False)
            if sname:
                k = SSpp.index(sname) + Mf
                l = IIp.index((iname, pname))
                
                circuit = circuit.feedback(k,l)
                SSpp.remove(sname)
                IIp.remove((iname, pname))
        
        SIGNAL = object()
        OO_effective = OOp + [(SIGNAL, s) for s in SSpp]
        II_effective = IIp + [(SIGNAL, s) for s in SSp]
        
        
        omapping = {}
        # construct output permutation
        for i, (iname, pname) in enumerate(OO_effective):
            if iname is not SIGNAL:
                eport = self.global_out.get((iname, pname), False)
            else:
                eport = self.signal_to_global_out.get(pname, False)
            if eport:
                omapping[i] = list(self.entity.out_port_identifiers).index(eport)
        
        imapping = {}
        # construct output permutation
        
        for i, (iname, pname) in enumerate(II_effective):
            if not (iname is SIGNAL):
                eport = self.global_in.get((iname, pname), False)
            else:
                eport = self.signal_to_global_in.get(pname, False)
            if eport:
                k = list(self.entity.in_port_identifiers).index(eport)
                imapping[k] = i
#        print(imapping, II_effective,self.signal_to_global_in)
        
        circuit = ca.map_signals_circuit(omapping, circuit.cdim) << circuit << ca.map_signals_circuit(imapping, circuit.cdim)

        self._circuit_data = circuit, circuit_symbols, self.instance_assignments
        self.entity.cdim = circuit.cdim
        return self._circuit_data



        
        
        
    def __repr__(self):
        return "%s('%s', '%s', %r, %r, %r)" \
                % (self.__class__.__name__, self.identifier, self.entity, self.components, self.signals, self.instance_assignments)
                


    
    def to_qhdl(self, tab_level = 0):
        
        components_qhdl = "\n".join([c.to_qhdl(tab_level = tab_level + 1) for c in self.components.values()])
        signals_qhdl = "    signal %s: fieldmode;\n" % ", ".join(self.signals.keys())
        format_map = lambda dd: ", ".join(["%s=>%s" % mm for mm in dd.items()])
        
        format_ass = lambda name, cname, generic_map, port_map : \
                            "    %s: %s %s %s;" % (name, cname,\
                                                ("" if not len(generic_map) \
                                                    else "generic map(%s);\n" % format_map(generic_map)),\
                                                    ("port map(%s)\n" % format_map(generic_map)))
        
        ret_str = """
architecture %s of %s is %s %s
    begin
        %s
    end architecture %s;
""" % (self.identifier, self.entity.identifier,
        components_qhdl,
        signals_qhdl,
        "\n     ".join([format_ass(a, *v) for a,v in self.instance_assignments.items()]),
        self.identifier)
        
        return ("\t"*tab_level) + ret_str.replace('\n', "\n"+ ("\t"*tab_level))
    
    



if __name__ == "__main__":
    from test.test_qhdl import *
    unittest.main()
            
                
