# coding=utf-8
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
The PLY-based QHDLParser class.
"""

from qnet.misc.parser import Parser, ParsingError
from qnet.qhdl import qhdl



class QHDLParser(Parser):
    
    def parse(self, inputstring):
        self.entities = {}
        self.architectures = {}
        return Parser.parse(self, inputstring)
        
    def create_circuit_lib(self, arch_id = None):
        if arch_id == None:
            if len(self.architectures) > 1:
                raise Exception('Multiple architectures found: %s. \
                        Please provide an arch_id argument selecting one architecture' % \
                        ", ".join(self.architectures.keys()))
            arch_id = self.architectures.keys().pop()

        arch = self.architectures[arch_id]
        
        arch_circuit = arch.to_circuit()
        print(arch_circuit)
        
    
    reserved = {
        'begin': 'BEGIN',
        'end': 'END',
        'entity': 'ENTITY',
        'component': 'COMPONENT',
        'architecture': 'ARCHITECTURE',
        'of': 'OF',
        'is': 'IS',
        'port': 'PORT',
        'in': 'IN',
        'out': 'OUT',
        'inout': 'INOUT',
        'fieldmode': 'FIELDMODE',
        'lossy_fieldmode': 'LOSSY_FIELDMODE',
        'generic': 'GENERIC',
        'signal': 'SIGNAL',
        'map': 'MAP',
        'real': 'REAL',
        'complex': 'COMPLEX',
        'int': 'INT'
    }

    tokens = list(reserved.values()) + [
        # Literals (identifier, integer constant, float constant, string constant, char const)
        'ID',
        'ICONST', 'FCONST',
        # Assignment ( :=, =>, <=)
        'ASSIGN', 'FEEDRIGHT','FEEDLEFT',
        # Delimeters ( ) , ; :
        'LPAREN', 'RPAREN',
        'COMMA', 'SEMI', 'COLON',
        ]


    def __init__(self, **kw):

        Parser.__init__(self, **kw)

        self.entities = {}
        self.architectures = {}    
        
    t_ignore = ' \t\x0c'

    # Newlines
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    # Assignment operators
    t_ASSIGN    = r':='
    t_FEEDRIGHT = '=>'
    t_FEEDLEFT = '<='
    # Delimeters
    t_LPAREN    = r'\('
    t_RPAREN    = r'\)'
    t_COMMA     = r','
    t_SEMI      = r';'
    t_COLON     = r':'

    # Identifiers and reserved words
    def t_ID(self, t):
        r"""[_A-Za-z][\w_]*"""
        t.type = self.reserved.get(t.value.lower(),"ID")
        return t
    # Integer literal
    t_ICONST = r'-?\d+' #([uU]|[lL]|[uU][lL]|[lL][uU])?'
    # Floating literal
    t_FCONST = r'-?((\d+)(\.\d+)(e(\+|-)?(\d+))? | (\d+)e(\+|-)?(\d+))' #([lL]|[fF])?'

    # Comments
    def t_comment(self, t):
        r"""--[^\n]*"""

    def t_error(self, t):
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)
        
    start = 'top_level_list'            
    
    def p_top_level_list(self, p):
        """
        top_level_list : top_level_list top_level_unit
                       | top_level_unit
        """
        if len(p) == 2:
            if isinstance(p[1], qhdl.Architecture):
                p[0] =  {'architectures':{p[1].identifier: p[1]}, 'entities': {}}

            elif isinstance(p[1], qhdl.Entity):
                p[0] = {'entities':{p[1].identifier: p[1]}, 'architectures': {}}
            else:
                raise ParsingError()
        else:
            if isinstance(p[2], qhdl.Architecture):
                assert p[2].identifier not in p[1]['architectures']
                p[1]['architectures'][p[2].identifier] = p[2]
                p[0] = p[1]

            elif isinstance(p[1], qhdl.Entity):
                assert p[2].identifier not in p[1]['entities']
                p[1]['entities'][p[2].identifier] = p[2]
                p[0] = p[1]
            else:
                raise ParsingError()
        
        
        
    def p_top_level_unit(self, p):
        """
        top_level_unit : entity_declaration 
                       | architecture_declaration
        """
        p[0] = p[1]
        
    def p_entity_declaration(self, p):
        """
        entity_declaration : ENTITY ID IS generic_clause port_clause END opt_entity opt_id SEMI
        """
        identifier = p[2]
        second_identifer = p[8]
        generics = p[4]
        ports = p[5]
        if second_identifer != None and identifier != second_identifer:
            raise ParsingError('IDs don\'t match: %s, %s' % (identifier, second_identifer))

        p[0] = self.entities[identifier] = qhdl.Entity(identifier, generics, ports)
    
    def p_opt_entity(self, p):
        """
        opt_entity : ENTITY
                   | empty
        """
        p[0] = p[1]
    
    def p_opt_id(self, p):
        """
        opt_id : ID
               | empty
        """
        p[0] = p[1]

    def p_opt_semi(self, p):
        """
        opt_semi : SEMI
                 | empty
        """
        pass

    def p_generic_clause(self, p):
        """
        generic_clause : generic_statement
                       | empty
        """
        if p[1] == None:
            p[0] = []
        else:
            p[0] = p[1]

    def p_empty(self, p):
        """ empty : """
        p[0] = None



    def p_generic_statement(self, p):
        """
        generic_statement : GENERIC LPAREN generic_list opt_semi RPAREN SEMI
        """
        p[0] = p[3]

    def p_generic_list(self, p):
        """
        generic_list : generic_list SEMI generic_entry_group
                     | generic_entry_group
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_generic_entry_group(self, p):
        """
        generic_entry_group : id_list COLON generic_type generic_default
        """
    
        p[0] = p[1], p[3], p[4]

    def p_id_list(self, p):
        """
        id_list : id_list COMMA ID
                | ID
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]
    

    def p_generic_type(self, p):
        """ generic_type : REAL
                         | COMPLEX 
                         | INT """
        p[0] = p[1]

    def p_generic_default(self, p):
        """
        generic_default : ASSIGN number
                        | empty
        """
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    def p_number(self, p):
        """
        number : simple_number
               | complex
        """
        p[0] = p[1]

    def p_simple_number(self, p):
        """
        simple_number : int
                      | real
        """
        p[0] = p[1]

    def p_int(self, p):
        """
        int : ICONST
        """
        p[0] = int(p[1])

    def p_real(self, p):
        """
        real : FCONST
        """
        p[0] = float(p[1])
    

    def p_complex(self, p):
        """
        complex : LPAREN  simple_number COMMA simple_number RPAREN
        """
        p[0] = complex(float(p[2]), float(p[4]))

    def p_port_clause(self, p):
        """
        port_clause : port_statement
                    | empty
        """
        if p[1] == None:
            p[0] = []
        else:
            p[0] = p[1]


    def p_port_statement(self, p):
        """
        port_statement : PORT LPAREN port_list opt_semi RPAREN SEMI
        """
        p[0] = p[3]

    def p_port_list(self, p):
        """
        port_list : with_io_port_list
                  | non_io_port_list
        """
#        if len(p) == 2:
        p[0] = p[1]
#        else:
#            p[0] = p[1] + [p[3]]


    def p_with_io_port_list(self, p):
        """
        with_io_port_list : io_port_entry_group SEMI non_io_port_list 
                          | io_port_entry_group
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3]
            
    def p_non_io_port_list(self, p):
        """
        non_io_port_list : non_io_port_entry_group SEMI non_io_port_list 
                          | non_io_port_entry_group
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3]


    def p_non_io_port_entry_group(self, p):
        """
        non_io_port_entry_group : id_list COLON signal_direction signal_type
        """
        p[0] = p[1], p[3], p[4]

    def p_io_port_entry_group(self, p):
        """
        io_port_entry_group : id_list COLON INOUT signal_type
        """
        p[0] = p[1], p[3], p[4]        


    def p_signal_direction(self, p):
        """
        signal_direction : IN
                         | OUT
        """
        p[0] = p[1]

    def p_signal_type(self, p):
        """
        signal_type : FIELDMODE 
                    | LOSSY_FIELDMODE
        """
        p[0] = p[1]

    def p_architecture_declaration(self, p):
        """
        architecture_declaration : ARCHITECTURE ID OF ID IS architecture_head BEGIN instance_mapping_assignment_list feedleft_assignment_list END opt_arch opt_id SEMI
        """
        a_id, e_id = p[2], p[4]
        component_list, signal_list = p[6]
        assignments = p[8]
        fl_assignments = p[9]
        if p[12] != None and p[12] != a_id:
            raise ParsingError('IDs don\'t match!')
        entity = self.entities.get(e_id, False)
        if not entity:
            raise qhdl.QHDLError('Entity %s not found.' % e_id)
        p[0] = self.architectures[a_id] = qhdl.Architecture(a_id, entity, component_list, signal_list, assignments, fl_assignments)


    def p_architecture_head(self, p):
        """
        architecture_head : component_declaration_list signal_list
        """
        p[0] = (p[1], p[2])
        
    def p_opt_arch(self, p):
        """
        opt_arch : ARCHITECTURE
                 | empty
        """
        p[0] = p[1]


    def p_component_declaration_list(self, p):
        """
        component_declaration_list : component_declaration_list component_declaration
                                   | component_declaration
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]


    def p_component_declaration(self, p):
        """
        component_declaration : COMPONENT ID generic_clause port_clause END COMPONENT opt_id SEMI
        """
        if p[7] != None and p[7] != p[2]:
            raise ParsingError('IDs don\'t match')
        p[0] = qhdl.Component(p[2], p[3], p[4])
        

    def p_signal_list(self, p):
        """
        signal_list : signal_list signal_entry_group
                    | signal_entry_group
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]


    def p_signal_entry_group(self, p):
        """
        signal_entry_group : SIGNAL id_list COLON signal_type SEMI
        """
    
        p[0] = p[2], p[4]
        
        

    def p_instance_mapping_assignment_list(self, p):
        """
        instance_mapping_assignment_list : instance_mapping_assignment_list instance_mapping_assignment
                                         | instance_mapping_assignment
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_instance_mapping_assignment(self, p):
        """
        instance_mapping_assignment : ID COLON ID generic_map port_map
        """
        p[0] = (p[1], p[3], p[4], p[5])

    def p_generic_map(self, p):
        """
        generic_map : GENERIC MAP LPAREN feedright_generic_assignment_list RPAREN SEMI
                    | empty
        """
        if len(p) == 2:
            p[0] = {}
        else:
            p[0] = p[4]

    def p_feedright_generic_assignment_list(self, p):
        """
        feedright_generic_assignment_list : feedright_generic_assignment_list COMMA feedright_generic_assignment
                             | feedright_generic_assignment
        """
        if len(p) == 2:
            asm = p[1]
            if isinstance(asm, dict):
                p[0] = asm
            else:
                p[0] = [p[1]]
        else:
            asm_list = p[1]
            asm = p[3]
            if isinstance(asm, dict):
                if not isinstance(asm_list, dict):
                    raise Exception("Either specify ALL assignments as 'from_id=>to_id|number' OR just as 'to_id|number'.")
                if len(set(asm.keys()) & set(asm_list.keys())) > 0:
                    raise Exception("Every from_id may only occur once!")
                asm_list.update(asm)
                p[0] = asm_list
            else:
                if not isinstance(asm_list, list):
                    raise Exception("Either specify ALL assignments as 'from_id=>to_id|number' OR just as 'to_id|number'.")
                p[0] = p[1] + [p[3]]

    def p_id_or_value(self, p):
        """
        id_or_value : ID
                   | number
        """
        p[0] = p[1]

    def p_feedright_generic_assignment(self, p):
        """
        feedright_generic_assignment : ID FEEDRIGHT id_or_value
                                     | id_or_value
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = {p[1]: p[3]}

    def p_feedright_port_assignment_list(self, p):
        """
        feedright_port_assignment_list : feedright_port_assignment_list COMMA feedright_port_assignment
                             | feedright_port_assignment
        """
        if len(p) == 2:
            asm = p[1]
            if isinstance(asm, dict):
                p[0] = asm
            else:
                p[0] = [p[1]]
        else:
            asm_list = p[1]
            asm = p[3]
            if isinstance(asm, dict):
                if not isinstance(asm_list, dict):
                    raise Exception("Either specify ALL assignments as 'from_id=>to_id' OR just as 'to_id'.")
                if len(set(asm.keys()) & set(asm_list.keys())) > 0:
                    raise Exception("Every from_id may only occur once!")
                asm_list.update(asm)
                p[0] = asm_list
            else:
                if not isinstance(asm_list, list):
                    raise Exception("Either specify ALL assignments as 'from_id=>to_id' OR just as 'to_id'.")
                p[0] = p[1] + [p[3]]

    def p_feedright_port_assignment(self, p):
        """
        feedright_port_assignment : ID FEEDRIGHT ID
                                  | ID
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = {p[1]: p[3]}

    def p_port_map(self, p):
        """
        port_map : PORT MAP LPAREN feedright_port_assignment_list RPAREN SEMI
                 | empty
        """
        if len(p) == 2:
            p[0] = {}
        else:
            p[0] = p[4]
    
    def p_feedleft_assignment_list(self, p):
        """
        feedleft_assignment_list : feedleft_assignment_list feedleft_assignment
                                 | feedleft_assignment
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            dkeys = set(p[1].keys()) & set(p[2].keys())
            if len(dkeys) != 0:
                raise Exception("Duplicate assignment of global identifiers %s" % (tuple(dkeys),))
            p[1].update(p[2])
            p[0] = p[1]
    
    def p_feedleft_assignment(self, p):
        """
        feedleft_assignment : ID FEEDLEFT ID SEMI
                            | empty
        """
        if len(p) == 2:
            p[0] = {}
        else:
            p[0] = {p[3]:p[1]}
        

    def p_error(self, p):
        print(p, self.lexer.lineno)
        raise ParsingError()

