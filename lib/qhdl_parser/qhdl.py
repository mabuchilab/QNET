#!/usr/bin/env python
# encoding: utf-8
"""
qhdl.py

This file contains the code to convert a circuit specified in QHDL into a Gough-James circuit expression.

The other file in this module 'qparse.py' takes care of the actual parsing of the qhdl source, while this file then
converts the netlist information into a circuit expression.

QHDL-file syntax:

entity my_first_entity is
    [generic ( var1: generic_type [:= default_var1]] [; var2: generic_type [...] ...]);]
    port (i_1,i_2,...i_n:in fieldmode; o_1,o_2,...o_n:out fieldmode);
end [entity] [my_first_entity];

architecture my_first_architecture of my_first_entity is
    component my_component
        [generic ( var3: generic_type [:= default_var3]] [; var4: generic_type [...] ...]);]
        port (p1,p2,...pm:in fieldmode; q1,q2,...qm:out fieldmode);
    end component [my_component];
    
    [component my_second_component
        [generic ( var5: generic_type [:= default_var5]] [; var6: generic_type [...] ...]);]
        port (p1,p2,...pr:in fieldmode; q1,q2,...qr:out fieldmode);
    
    end component [my_second_component];
    
    ...
    
    ]
    
    [signal s_1,s_2,s_3,...s_m fieldmode;]

begin
    COMPONENT_INSTANCE_ID1: my_component
        [generic map(var1 => var3, var1 => var4);]
        port map (i_1, i_2, ... i_m, s_1, s_2, ...s_m);
    
    [COMPONENT_INSTANCE_ID2: my_component
        [generic map(var1 => var3, var1 => var4);]
        port map (s_1, s_2, ... s_m, o_1, o_2, ...o_m);
    
    COMPONENT_INSTANCE_ID3: my_second_component
        [generic map (...);]
        port map (...);
    
    ...
        
        ]
end [architecture] [my_first_architecture];



generic_type = int | real | complex


"""

def my_debug(msg):
    pass
    # print msg


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
        g_str = "generic (%s);\n" % ("; ".join(["%s: %s%s" % (identifier, gtype, default_str(default)) \
                                                for identifier,(gtype, default) in self.generics.items()]))
        return (tab_level*"\t") + g_str
    
    def ports_to_qhdl(self, tab_level):
        
        in_str = "%s: in fieldmode" % ", ".join(self.port_identifiers[ : self.cdim/2])
        out_str = "%s: out fieldmode" % ", ".join(self.port_identifiers[self.cdim/2 : ])
        p_str = "port (%s; %s);" % (in_str, out_str)
        
        return (tab_level*"\t") + p_str
        
    
    def __init__(self, identifier, generics, ports):
        self.identifier = identifier
        
        self.generics = {}
        self.in_ports = {}
        self.out_ports = {}
        self.ports = {}
        port_identifiers = []
        generic_identifiers = []
        self.in_port_identifiers = []
        self.out_port_identifiers = []

        
        for (identifier_list, gtype, default_val) in generics:
            if gtype not in GENERIC_TYPES:
                raise QHDLError("Generic type of generics %s is %s, but must be one of the following: %r" \
                                         % (", ".join(identifier_list), gtype, GENERIC_TYPES))
            for gid in identifier_list:
                if gid in self.generics:
                    raise QHDLError("Generic identifier non-unique: %s" % identifier)
                
                self.generics[gid] = gtype, default_val
                generic_identifiers.append(gid)
        
        for (identifier_list, direction, signal_type) in ports:
            if direction not in ('in', 'out'):
                raise QHDLError(str((identifier_list, direction, signal_type)))
            
            if signal_type != "fieldmode":
                raise NotImplementedError
            for pid in identifier_list:
                if pid in self.generics:
                    raise QHDLError("There already exists a generic with the same identifier : %s" % pid)
                if pid in self.ports:
                    raise QHDLError("Port identifier non-unique: %s" % pid)
                
                if direction == 'in':
                    pindex = len(self.in_ports)
                    self.in_ports[pid] = pindex
                    self.in_port_identifiers.append(pid)
                else:
                    pindex = len(self.out_ports)
                    self.out_ports[pid] = pindex
                    self.out_port_identifiers.append(pid)
                self.ports[pid] = direction, pindex
                
                port_identifiers.append(pid)
        
        if not len(self.in_ports) == len(self.out_ports):
            raise QHDLError('The numbers of input and output channels do not match.')
        self._port_identifiers = tuple(port_identifiers)
        self._generic_identifiers = tuple(generic_identifiers)
                                    
    
    @property
    def cdim(self):
        return len(self.in_ports)
    
    def port_identifiers():
        doc = "The port_identifiers property."
        def fget(self):
            return self._port_identifiers
        return locals()
    pids = port_identifiers = property(**port_identifiers())
    
    def generic_identifiers():
        doc = "The generic_identifiers property."
        def fget(self):
            return self._generic_identifiers
        return locals()
    gids = generic_identifiers = property(**generic_identifiers())
        
    
    def __getitem__(self, key):
        if key in self.ports:
            return self.ports[key]
        elif key in self.generics:
            return self.generics[key]
        else:
            raise KeyError(str(key))
    
    def get(self, key, default_val = None):
        try:
            return self[key]
        except KeyError:
            return default_val
    
    def has_generic(self, gid):
        return gid in self.generics
    
    def has_port(self, pid):
        return pid in self.ports
        
        

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
    
    def __init__(self, identifier, entity, components, signals, assignments, global_assignments = {}):
        self.identifier = identifier
        self.entity = entity
        self.components = {}
        self.signals = {}
        self.instance_assignments = {}
        self._circuit = False
        
        # lookuptables format
        # (instance_name, instance_port_name)
        #       => ((target_instance_name|'entity'), port_name, port_id, (connecting_signal_name|None))
        mediated_inport_map = {}
        mediated_outport_map = {}
        
        #process components
        for component in components:
            #check for duplicate component identifiers
            if component.identifier in self.components:
                raise QHDLError('Component ID non-unique: %s' % component.identifier)
            self.components[component.identifier] = component
        
        #process signals
        for signal_ids, signal_type in signals:
            if signal_type != 'fieldmode':
                raise NotImplementedError
            
            for sid in signal_ids:
                #check if signal identifier coincides with entity port name
                if sid in entity.port_identifiers:
                    raise QHDLError('Signal identifier already used as an entity port identifier: %s' % sid)
                
                #check for duplicate signal identifiers
                if sid in self.signals:
                    raise QHDLError('Signal identifier non-unique: %s' % sid)
                #every signal can only connect two ports of component instances,
                #one in-port and one out-port
                self.signals[sid] = {'into': None, 'outfrom': None}
        
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
                for gname, (gtype, default) in component.generics.items():
                    if default is None:
                        raise QHDLError(('No default value defined for generic %s of component %s,\n' % (gname, component.identifier)) 
                                    + 'either define this or add a generic map statement to the instance assigment of %s' % (instance_name,))
                
            
            #if generic map is already in explicit form, assert that the
            #assigned q,r,... coincide those defined in the component
            elif sorted(generic_map.keys()) != sorted(component.generics.keys()):
                raise QHDLError('All generics of component %s must be assigned' % component.identifier)
            
            #make sure referenced a,b,c,... exist as generics in the entity
            for name_in_c, name_in_e in generic_map.items():
                entity_g = entity.generics.get(name_in_e, False)
                if not entity_g:
                    raise QHDLError('Entity %s does not define a generic of name %s' % (entity.identifier, name_in_e))
                
                component_g = component.generics.get(name_in_c, False)
                
                #probably redundant
                if not component_g:
                    raise QHDLError('Component %s does not define a generic of name %s' % (component.identifier, name_in_c))
                #check that generics have compatible type
                #e.g. component_generic_type == real, entity_generic_type == int is okay
                #because an int may be cast to a real, however the other way around does not always work!
                if not gtype_compatible(component_g[0], entity_g[0]):
                    raise QHDLError('Mapped generics are of incompatible type.' \
                                        + 'Change the generic\'s type in the entity to %s' % component_g[0])
            
            #convert port map (a,b,c,...) into port map (q=>a, r=>b,...)
            #based on the ports of the component
            if not isinstance(port_map, dict):
                if not len(port_map) == len(component.port_identifiers):
                    raise QHDLError('All ports of instance %s(%s) must be mapped' % (instance_name, component.identifier))
                port_map = dict(zip(component.port_identifiers, port_map))
            
            #if port map is already in explicit form, assert that the
            #assigned q,r,... coincide those defined in the component
            elif sorted(port_map.keys()) != sorted(component.ports.keys()):
                raise QHDLError('All ports of instance %s(%s) must be mapped' % (instance_name, component.identifier))
            
            #create lookup tables
            for name_in_c, name_in_e in port_map.items():
                
                #any referenced a,b,c,... must either exist
                #as a signal in the architecture or a port of the entity
                entity_p = entity.ports.get(name_in_e, False)
                signal = self.signals.get(name_in_e, False)
                if not entity_p and not signal:
                    raise QHDLError('The entity %s does not define a port\
                                    and the architecture %s\
                                    does not any define any signal of\
                                    name %s ' % (entity.identifier, self.identifier, name_in_e))
                                    
                
                component_p = component.ports.get(name_in_c, False)
                #probably redundant
                if not component_p:
                    raise QHDLError('Component %s does not define a port of name %s' % (component.identifier, name_in_c))
                
                c_dir, c_id = component_p
                
                #any signal may only connect two ports, one in-port and one out-port
                #if port is mapped to a signal, then on the other end of that
                if signal:
                    
                    #if port is ingoing
                    if c_dir == 'in':
                        
                        #assert that signal is not already connected to some other ingoing port
                        if signal['into'] != None:
                            raise QHDLError('Signal %s is connected to more than one ingoing ports' % name_in_e)
                        
                        signal['into'] = instance_name, name_in_c, c_id, component
                        
                        #if 'other end' of signal is already connected, add entries to lookup table
                        if signal['outfrom'] != None:
                            mediated_inport_map[(instance_name, name_in_c)] = signal['outfrom'] + (name_in_e,)
                            mediated_outport_map[(signal['outfrom'][0:2])] = instance_name, name_in_c, c_id, component, name_in_e
                    else:
                        #assert that signal is not already connected to some other outgoing port
                        if signal['outfrom'] != None:
                            raise QHDLError('Signal %s is connected to more than one outgoing ports' % name_in_e)
                        signal['outfrom'] = instance_name, name_in_c, c_id, component
                        
                        #if 'other end' of signal is already connected, add entries to lookup table
                        if signal['into'] != None:
                            mediated_outport_map[(instance_name, name_in_c)] = signal['into'] + (name_in_e,)
                            mediated_inport_map[(signal['into'][0:2])] = instance_name, name_in_c, c_id, component, name_in_e
                
                else: #port is directly mapped to an entity port
                    e_dir, e_id = entity_p
                    
                    #entity port must have same direction
                    if c_dir != e_dir:
                        raise QHDLError('Component port %s.%s must be connected to entity port of same direction' % (component.identifier, name_in_c))
                    
                    #add lookup table entries
                    if c_dir == 'in':
                        mediated_inport_map[(instance_name, name_in_c)] = 'entity', name_in_e, e_id, entity, None
                        if ('entity', name_in_e) in mediated_outport_map:
                            raise QHDLError('The entity in-port %s.%s is connected to multiple instance ports' \
                                                        % (entity.identifier, name_in_e))
                        mediated_outport_map[('entity', name_in_e)] = instance_name, name_in_c, c_id, component, None
                    else: #c_dir == 'out'
                        mediated_outport_map[(instance_name, name_in_c)] = 'entity', name_in_e, e_id, entity, None
                        if ('entity', name_in_e) in mediated_inport_map:
                            raise QHDLError('The entity out-port %s.%s is connected to multiple instance ports' \
                                                        % (entity.identifier, name_in_e))
                        
                        mediated_inport_map[('entity', name_in_e)] = instance_name, name_in_c, c_id, component, None
                
                #store (redundant) information for book-keeping and string representations
                self.instance_assignments[instance_name] = component, generic_map, port_map
                
        for source_id, target_id in global_assignments.items():
            if target_id in self.entity.out_port_identifiers:
                if not source_id in self.signals:
                    raise QHDLError('Global Out-Ports may only be assigned to signals')
                outfrom = self.signals[source_id].get('outfrom', False)
                if not outfrom:
                    raise QHDLError('GLobal Out-Port can only be connected to instance out port')
                if self.signals[source_id].get('into', False):
                    raise QHDLError('Signal connects into several different ports')
                
                mediated_inport_map[('entity', target_id)] = outfrom + (None,)
                mediated_outport_map[outfrom[0:2]] = 'entity', target_id, self.entity.ports.get(target_id)[1], self.entity, None
                del self.signals[source_id]
                
            elif target_id in self.signals:
                if not source_id in self.entity.in_port_identifiers:
                    raise QHDLError('Signals can only be assigned to global In-Ports')
                into = self.signals[target_id].get('into', False)
                if not into:
                    raise QHDLError('GLobal In-port can only be connected to instance in-port')
                if self.signals[target_id].get('outfrom', False):
                    raise QHDLError('Signal connects into several different ports')
                
                mediated_outport_map[('entity', source_id)] = into + (None,)
                mediated_inport_map[into[0:2]] = 'entity', source_id, self.entity.ports.get(source_id)[1], self.entity, None
                del self.signals[target_id]
            else:
                raise QHDLError('Global Assignment Error')
                    
        
        #combine lookup tables
        self.mediated_port_map = {'in': mediated_inport_map, 'out': mediated_outport_map}
    
    def to_circuit(self, identifier_postfix = ''):
        if self._circuit:
            return self._circuit
        from algebra import circuit_algebra as ca
        
        #shortcut
        pm = self.mediated_port_map
        pmi = pm['in']
        pmo = pm['out']
        
        cdim = self.entity.cdim
        reached_entity = {}
        
        #initial circuit values
        current_circuit = ca.circuit_identity(cdim)
        
        current_channel = ca.circuit_identity(1)
        
        #create symbols for all instances
        circuit_symbols = {}
        for (instance_name, (component, generic_map, port_map)) in self.instance_assignments.items():
            circuit_symbols[instance_name] = ca.CSymbol(instance_name+identifier_postfix, component.cdim)
        
        
        #initial global in ports

        current_global_in_ports = [('entity', port_name) for port_name in self.entity.in_port_identifiers]
        # print 'initial current_global_in_ports', current_global_in_ports
        # unvisited_instances = self.instance_assignments.keys()
        complex_instance_stack = {}
        feedback_stack = {}
        feedback_stack_order = []
        blocked_channels = set([])
        
        
        while True:
            
            #iterate over all 'channels', i.e. current_global_in_ports
            for channel_index, (instance_name, port_name) in enumerate(current_global_in_ports):
                
                #a channel may be blocked from a previous iteration of the outer
                #while loop if an unfinished complex component is at its end
                if channel_index in blocked_channels or channel_index in reached_entity:
                    # my_debug("skipping channel index: %d" % channel_index)
                    continue
                
                reached_complex = False
                
                
                while(not reached_complex):
                    
                    (next_instance, next_port_name, next_port_id, next_component, signal) = pmo[(instance_name, port_name)]
                    # print instance_name, port_name , "->", next_instance, next_port_name
                    # a, b = pmi[(next_instance, next_port_name)][0:2]
                    # print  a,b, "->", next_instance, next_port_name
                    assert pmi[(next_instance, next_port_name)][0:2] == (instance_name, port_name)
                    
                    my_debug("in channel %d visited instance: %r" % (channel_index, (next_instance, next_port_name, next_port_id)))
                    # my_debug(blocked_channels)
                    
                    if next_instance == 'entity':
                        reached_entity[channel_index] = next_port_id
                        # print reached_entity, next_port_id
                        break
                    
                    if next_component.cdim == 1:
                        assert next_port_id == 0
                        
                        #replace global in port by the output port of this instance
                        current_global_in_ports[channel_index] = (instance_name, port_name) \
                                                                = (next_instance, next_component.out_ports.keys().pop())
                        
                        #calculate the next layer of circuit for the current instance
                        next_circuit = (ca.circuit_identity(channel_index) \
                                            + circuit_symbols[next_instance] \
                                            + ca.circuit_identity(cdim - channel_index - 1))
                        # print "="*100
                        # print next_circuit, "<<", current_circuit
                        #feed the existing system into the new layer
                        current_circuit =  next_circuit << current_circuit
                        # print current_circuit
                        # print "="*100 
                        my_debug("# processed instance %s(%s)" % (next_instance, next_component.identifier))
                    
                    else: #component.cdim >= 2
                        
                        reached_complex = True
                        
                        
                        
                
                if reached_complex:
                    my_debug("channel blocked by %s:%s" % (next_instance, next_port_name))
                    blocked_channels.add(channel_index)
                    if next_instance in feedback_stack:
                        if next_instance == feedback_stack_order[-1]:
                            
                            stored_current_circuit, P_inv_perm, feedback_channels, bypassed_channels, feedback_map = feedback_stack[next_instance]
                            
                            
                            # my_debug(permutation)
                            # my_debug(bypassed_channels)
                            # my_debug(next_port_id)
                            feedback_map[channel_index] = P_inv_perm[bypassed_channels + next_port_id]
                            
                            
                            
                            if len(feedback_map) == feedback_channels:
                                if len(complex_instance_stack) == 0:
                                    # inverted_permutation = ca.invert_permutation(permutation)
                                    
                                    # my_debug(feedback_stack_order)
                                    # my_debug(feedback_map)
                                    # my_debug(cdim)
                                    # my_debug(feedback_channels)
                                    # my_debug(current_global_in_ports)
                                    assert all((v >= cdim - feedback_channels for v in feedback_map.values()))
                                    
                                    
                                    #can close feedback loops!
                                    new_permutation = ca.map_signals(feedback_map, cdim)
                                    new_map_circuit = ca.CPermutation(new_permutation)
                                    
                                    current_circuit = new_map_circuit << current_circuit
                                    
                                    
                                    for k in xrange(feedback_channels):
                                        current_circuit = current_circuit.feedback()
                                    # print "="*100
                                    # print current_circuit, "<<", stored_current_circuit
                                    current_circuit = current_circuit << stored_current_circuit
                                    # print current_circuit
                                    # print "="*100

                                    
                                    blocked_channels = set()
                                    reached_entity = {}
                                    current_global_in_ports = [p for k,p in enumerate(current_global_in_ports) if k not in feedback_map]
                                                                        
                                    del feedback_stack[next_instance]
                                    
                                    feedback_stack_order.pop()
                                    
                                    cdim = cdim - feedback_channels
                                    
                                    break # restart iteration over current_global_in_ports because it was changed!
                                    
                                    my_debug("closed feedback loop of entity %s" % next_instance)
                                    my_debug("feedback stack %s" % feedback_stack)
                                    my_debug("blocked channels %s" % blocked_channels)
                                    my_debug("reached entity %s" % reached_entity.keys())
                                    my_debug("current global in ports %s" % current_global_in_ports)
                                    my_debug("next ports: %s" % [pmo[cgi][0:2] for cgi in current_global_in_ports])
                                    
                                else:
                                    my_debug("instances blocking feedback: %r" % feedback_stack.keys())
                                
                            else:
                                my_debug("instance too low on stack: %s" % next_instance)
                        continue
                            
                    
                    #handle case where next_instance is a new complex instance
                    elif next_instance not in complex_instance_stack:
                        complex_instance_stack[next_instance] = {channel_index: next_port_id}
                        # at this time the instance cannot be further handled, because only one port is connected
                    else:
                        
                        port_map = complex_instance_stack[next_instance]
                        
                        if channel_index in port_map and port_map[channel_index] != next_port_id:
                            raise Exception('Can only connect to target port %s.%s once!' % (next_instance, next_port_name))
                        
                        port_map[channel_index] = next_port_id
                        
                        
                        i_cdim = next_component.cdim
                        
                        
                        #if all incoming ports for the complex instance are resolved
                        if len(port_map) == i_cdim:
                            
                            #remove this instance from the complex instance stack
                            del complex_instance_stack[next_instance]
                            
                            #remove the blockade of this instance's channels
                            blocked_channels.difference_update(set(port_map.keys()))
                            
                            
                            """
                            ####### circuit  alignment #################################################################
                            channel index ################################### permutation # instance # inverse perm.
                             
                             0                                          ----------------------------------------
                                                                        ----------------------------------------
                                                                        ----------------------------------------
                                                                                      ...                    ...
                                                                        ----------------------------------------
                             min_port                                   ----    ---------   --------   ---------
                                                                        ----    |       |   |       |  |       |
                                                                        ----    |       |   | Inst. |  |    -1 |
                                                                        ----    | Perm. |   |       |  | Perm. |
                                                                        ...        ...         ...        ...
                                                                        ----    |       |   |       |  |       |
                             min_port + i_cdim - 1                 ----    |       |   ---------  |       |
                                                                        ----    |       |   ---------  |       |
                                                                        ----    |       |   ---------  |       |
                                                                        ----    |       |      ...     |       |
                             max_port                                   ----    ---------   ---------  ---------
                                                                        ----------------------------------------
                                                                        ----------------------------------------
                                                                                      ...                    ...
                             cdim-1                                ----------------------------------------
                            ################################################# permutation # instance # inverse perm.
                            """
                            assert sorted(port_map.values()) == range(i_cdim)
                            
                            
                            #construct permutation
                            min_port = min(port_map.keys())
                            max_port = max(port_map.keys())
                            permutation = []
                            
                            # keep track how many previous ports
                            # were not mapped to the instance ports
                            not_in_map_count = 0
                            
                            for p in range(min_port, max_port + 1):
                                
                                if p in port_map:
                                    permutation.append(port_map[p])
                                else:
                                    #the relative order of bypassed channels is not changed!
                                    #p is mapped to i_cdim + not_in_map_count (+ min_port)
                                    permutation.append(i_cdim + not_in_map_count)
                                    not_in_map_count += 1
                            
                            permutation = tuple(permutation)
                            
                            # my_debug(next_instance, permutation)
                            
                            assert sorted(permutation) == range(max_port - min_port + 1)
                            
                            #construct new current_circuit
                            instance_block = circuit_symbols[next_instance] \
                                                + ca.circuit_identity(max_port - min_port - i_cdim + 1)
                            
                            perm_instance_block = instance_block << ca.CPermutation(permutation)
                            
                            
                            #restore the original port-channel mapping
                            #for all unaffected (bypassed) ports
                            perm_instance_block = ca.CPermutation(ca.invert_permutation(permutation)) << perm_instance_block
                            
                            #update only the affected channels of the global in ports
                            c_out_port_map = next_component.out_port_identifiers
                            
                            for channel_index, instance_port_index in port_map.items():
                                current_global_in_ports[channel_index] = next_instance, c_out_port_map[instance_port_index]
                                
                            blocked_channels.difference_update(set(port_map.keys()))
                            blocked_channels = set()
                            
                            my_debug("# processed instance %s(%s)" % (next_instance, next_component.identifier))
                            
                            #full block including the padding below and above
                            next_circuit = ca.circuit_identity(min_port) \
                                                + perm_instance_block \
                                                        + ca.circuit_identity(cdim - max_port - 1)
                            #feed full system into block
                            # print "="*100
                            # print next_circuit, "<<", current_circuit
                            current_circuit = next_circuit << current_circuit
                            # print current_circuit
                            # print "="*100                            
            
                            
            
            if len(reached_entity) == self.entity.cdim and len(feedback_stack) == 0:
                break
            
            assert len(reached_entity) <= self.entity.cdim
            
            if len(blocked_channels) + len(reached_entity) == cdim:
                # there must be a feedback loop

                #take the first entry of the complex instance stack
                feedback_instance = complex_instance_stack.keys().pop()
                
                my_debug("pushing %s onto feedback stack" % feedback_instance)
                
                
                feedback_port_map = complex_instance_stack[feedback_instance]
                # print "="*100
                # print "FEEDBACK PORT MAP"
                # print "="*100                
                # print feedback_port_map
                # print "="*100
                
                feedback_component = self.instance_assignments[feedback_instance][0]
                
                #number of channels that require a feedback loop
                feedback_channels = feedback_component.cdim - len(feedback_port_map)
                
                # print cdim
                #number of channels to be bypassed 'above' the component
                bypassed_channels = cdim - len(feedback_port_map)

            

                #                  ___________               ___________                ___________
                #      --->--------|         |---------------|         |-         ------|         |-------->------
                #      --->--------|         |---------------|         |-         ------|         |-------->------
                #                               ...                                                          
                #      --->--------|    P    |---------------|  P^-1   |-         ------|    P*   |-------->------
                #                               ___________                                                      
                #      --->--------|         |--|         |--|         |-   ....  ------|         |-------->------
                #                      ...          ...          ...                        ...                  
                #      --->--------|         |--|         |--|         |-         ------|         |-------->------
                #          /-------|         |--|         |--|         |-         ->----|         |--------\ 
                #          |           ...          ...          ...                        ...            |
                #          | /-----|_________|--|_________|--|_________|-         ->----|_________|-----\  |
                #          | |                                                                          |  |
                #          | \--------------------------------------------------------------------------/  |
                #          \-------------------------------------------------------------------------------/
                #
                
                # permute channels such that feedback_instance extends outwards below main channels with all virtual channels
                # cf. P in the figure in the comment above
                mapping = dict(((channel, port_index + bypassed_channels) for channel, port_index in feedback_port_map.items()))
                
                
                # print "="*100
                # print "bypassed channels, mapping"
                # print "="*100                
                # print bypassed_channels, mapping
                # print "="*100
                
                
                
                # this also guarantees that all of the added feedback_channels are mapped to channels that
                # feed into the component
                P_perm = ca.map_signals(mapping, cdim + feedback_channels)
                # print P_perm
                
                P = ca.CPermutation(P_perm)
                # print P
                
                feedback_mapping = dict(((k, P_perm[k + cdim] - bypassed_channels) \
                                            for k in xrange(feedback_channels)))
                                            
                assert all((v >= 0 for v in feedback_mapping.values()))
                                
                P_inv_perm = ca.invert_permutation(P_perm)
                try:
                    last_item_on_fstack = feedback_stack_order[-1]
                    #reset channel mapping because it could change after closing this feedback loop
                    feedback_stack[last_item_on_fstack][4].clear()
                except IndexError:
                    pass
                    
                # print "="*100
                # print "CIRCUIT BEFORE FEEDBACK"                
                # print "="*100
                # print current_circuit
                # print "="*100
                feedback_stack[feedback_instance] = (current_circuit, P_inv_perm, feedback_channels, bypassed_channels, {})                
                feedback_stack_order.append(feedback_instance)
                
                current_circuit = P.series_inverse() << (ca.circuit_identity(bypassed_channels) + circuit_symbols[feedback_instance]) << P
                # print "="*100
                # print "NEW INITIAL Circuit WITHIN FEEDBACK LOOP"
                # print "="*100
                # print current_circuit
                # print "="*100
                
                for k in xrange(cdim):
                    if k in feedback_port_map:
                        current_global_in_ports[k] = (feedback_instance, feedback_component.out_port_identifiers[feedback_port_map[k]])
                
                for k in xrange(feedback_channels):
                    current_global_in_ports.append((feedback_instance, \
                                                    feedback_component.out_port_identifiers[feedback_mapping[k]]))
                cdim += feedback_channels
                del complex_instance_stack[feedback_instance]
                ####
                # complex_instance_stack.clear()
                ####
                # for k in feedback_port_map:
                #     blocked_channels.remove(k)
                # blocked_channels.difference_update(set(feedback_port_map.keys()))
                blocked_channels = set()
                reached_entity = {}
        
        entity_map = ca.map_signals_circuit(reached_entity, cdim)
        self._circuit =  entity_map << current_circuit, circuit_symbols, self.instance_assignments
        return self._circuit

    
    
    
    # def to_circuit(self, identifier_postfix = ''):
        
        
        
        
    def __repr__(self):
        return "%s('%s', '%s', %r, %r, %r)" \
                % (self.__class__.__name__, self.identifier, self.entity, self.components, self.signals, self.instance_assignments)
                


    
    def to_qhdl(self, tab_level = 0):
        components_qhdl = "\n".join([c.to_qhdl(tab_level = tab_level + 1) for c in self.components.values()])
        signals_qhdl = "    signal %s: fieldmode;\n" % ", ".join(self.signals.keys())
        format_map = lambda dd: ", ".join(["%s=>%s" % mm for mm in dd.items()])
        
        format_ass = lambda name, (cname, generic_map, port_map) : \
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
        "\n     ".join([format_ass(a, v) for a,v in self.instance_assignments.items()]),
        self.identifier)
        
        return ("\t"*tab_level) + ret_str.replace('\n', "\n"+ ("\t"*tab_level))
    
    def to_dot_graph(self, prefix = ''):
        in_node = """
    {node [shape=record label="in | %s"] %sinput}
    {rank=source; %sinput}""" \
            % (" | ".join(["<%s>%s" % (p_name, p_name) for  p_name in self.entity.in_port_identifiers]),\
                prefix, prefix)
        
        out_node = """
    {node [shape=record label="out | %s"] %soutput}
    {rank=sink; %soutput}""" \
            % (" | ".join(["<%s>%s" % (p_name, p_name) for p_name in self.entity.out_port_identifiers]),\
                prefix, prefix)
        
        intermediate_nodes = ["{node [shape=record label=\"{{%s}|{%s (%s)}|{%s}}\"] %s;}" \
            %  (" | ".join(["<%s>%s" % (p_name, p_name) for p_name in component.in_port_identifiers]),\
                instance, component.identifier,\
                " | ".join(["<%s>%s" % (p_name, p_name) for p_name in component.out_port_identifiers]),\
                prefix + instance) for instance, (component, generic_map, port_map) in self.instance_assignments.items()]
        
        edges = []

        
        for (source_instance, ns_i), (target_instance, nt_i, k_i, c, s) in self.mediated_port_map['out'].items():
            if source_instance =="entity":
                edges.append("%sinput:%s -> %s:%s" % (prefix, ns_i, prefix + target_instance, nt_i))
            elif target_instance == 'entity':
                edges.append("%s:%s -> %soutput:%s" % (prefix + source_instance, ns_i, prefix, nt_i))
            else:
                edges.append("{edge [label = \"%s\"] %s:%s -> %s:%s}" % (s, prefix + source_instance, ns_i, prefix + target_instance, nt_i))
        
        # for (source_instance, ns_i), (target_instance, nt_i, k_i, c, s) in self.mediated_port_map['in'].items():
        #     if source_instance =="entity":
        #
        #         edges.append("{ edge in_ports:%s -> %s:%s}" % (ns_i, target_instance, nt_i))
        #     else:
        #         edges.append("{ edge [label = \"%s\"] %s:%s -> %s:%s}" % (s, source_instance, ns_i, target_instance, nt_i))
                
                
        
        dot = """digraph %s {
	graph [center rankdir=LR nodesep=.5 ranksep=.5 label="architecture %s of entity %s"]
	edge [shape=normal]
	%s
	
	%s
	%s
	%s
}"""    % (prefix + "layout", self.identifier, self.entity.identifier, in_node, "\n\t".join(intermediate_nodes), out_node, "\n\t".join(edges))
        return dot
    
    def save_dot_graph(self, filename, overwrite = True):
        
        if not filename.endswith('.dot') or filename.endswith('.gv'):
            filename += ".gv"
        from os import path
        if path.exists(filename):
            if not overwrite:
                my_debug("File already exists. Aborting")
                return
            
            if not path.isfile(filename):
                raise Exception('The file %s already exists but it is not a regular file.')
        with open(filename, 'w') as dot_file:
            dot_file.write(self.to_dot_graph())
        my_debug("output written to draphviz-file")





        
                
