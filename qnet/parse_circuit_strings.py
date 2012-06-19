#!/usr/bin/env python
# encoding: utf-8
"""
parse_circuit_strings.py

Created by Nikolas Tezak on 2011-05-13.
Copyright (c) 2011 . All rights reserved.

Parse strings into Circuit expressions.
See documentation for 'parse_circuit_string()'
"""


import algebra.circuit_algebra as ca

def parse_circuit_string(circuit_string): 
    """
    Parse strings for symbolic Circuit expressions into actual expression objects.
    
    A string may contain multiple subsequent expressions. In any case a list of all parsed expressions is returned.
    
    Examples:
        'a_nice_name(3)' --> CSymbol('a_nice_name', 3)
        'a(3) + b(5)' -> Concatenation(CSymbol('a', 3), CSymbol('b', 5))
        'a(3) << b(3)' -> SeriesProduct(CSymbol('a', 3), CSymbol('b', 3))
        '(a(3) + cid(1)) << b(4)' -> SeriesProduct(Concatenation(CSymbol('a', 3), CIdentity()), CSymbol('b', 3))
        '[a(5)]_(1->2)' -> Feedback(CSymbol('a', 5), 1, 2)
    """
    p = CircuitExpressionParser()
    return p.parse(circuit_string)
    

from qhdl_parser.qparse import Parser
class CircuitExpressionParser(Parser):
    
    def parse(self, inputstring):
        self.entities = {}
        self.architectures = {}
        return Parser.parse(self, inputstring)

        
    
    reserved = {
        'cid': 'CID',
        'fb': 'FB',
        'p_sigma': 'PSIGMA',
    }

    tokens = reserved.values() + [
        # Literals: identifier, integer constant
        'ID', 'ICONST',
        # Operations +, <<, >>
        'PLUS', 'LSERIES', # 'RSERIES',
        # Arrow ( -> )
        'RARROW',
        # Delimeters ( ) [ ] _ ,
        'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'UNDER', 'COMMA',
        ]


    def __init__(self, **kw):
        Parser.__init__(self, **kw)
 
        
    t_ignore = ' \t\x0c'

    # Newlines
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    # Assignment operators
    t_PLUS    = r'\+'
    
    t_LSERIES = r'<<'
    # t_RSERIES = r'>>'

    # Delimeters
    t_LPAREN    = r'\('
    t_RPAREN    = r'\)'
    t_LBRACKET  = r'\['
    t_RBRACKET  = r'\]'    
    t_UNDER     = r'_'
    t_RARROW    = r'\->'
    t_COMMA     = r','

    # Identifiers and reserved words
    def t_ID(self, t):
        r'[A-Za-z][\w_]*'
        t.type = self.reserved.get(t.value.lower(),"ID")
        return t
        
    # Integer literal
    t_ICONST = r'\d+'

    # Comments
    def t_comment(self, t):
        r'\#[^\n]*'

    def t_error(self, t):
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)
        
    start = 'expression_list'            
    
    def p_expression_list(self, p):
        """
        expression_list : expression_list circuit_expression
                       | circuit_expression
                       | 
        """
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        elif len(p) == 2:
            p[0] = [p[1]]
        else: 
            p[0] = []
        
        
    def p_circuit_expression(self, p):
        """
        circuit_expression : closed_expression
                       | circuit_concatenation
                       | circuit_series
        """
        p[0] = p[1]
        
    def p_closed_expression(self, p):
        """
        closed_expression : bracketed_circuit_expression
                          | circuit_identity
                          | circuit_symbol
                          | circuit_feedback
                          | circuit_permutation
        """
        p[0] = p[1]

    def p_bracketed_circuit_expression(self, p):
        """
        bracketed_circuit_expression : LPAREN circuit_expression RPAREN
        """
        p[0] = p[2]
        
    def p_circuit_identity(self, p):
        """
        circuit_identity : CID LPAREN ICONST RPAREN
        """
        p[0] = ca.cid(int(p[3]))
        
    def p_circuit_symbol(self, p):
        """
        circuit_symbol : ID LPAREN int RPAREN
        """
        p[0] = ca.CSymbol(p[1], p[3])
    
    def p_circuit_concatenation_1(self, p):
        """
        circuit_concatenation : closed_expression PLUS closed_expression
        """
        p[0] = ca.Concatenation(p[1], p[3])
    
    def p_circuit_concatenation_2(self, p):
        """
        circuit_concatenation : closed_expression PLUS circuit_concatenation
        """
        p[0] = ca.Concatenation(p[1], *p[3].operands)
    
    
    def p_circuit_series_1(self, p):
        """
        circuit_series : closed_expression LSERIES closed_expression
        """
        p[0] = ca.SeriesProduct(p[1], p[3])

    def p_circuit_series_2(self, p):
        """
        circuit_series : closed_expression LSERIES circuit_series
        """
        p[0] = ca.SeriesProduct(p[1], *p[3].operands)
        
    
    def p_circuit_feedback_1(self, p):
        """
        circuit_feedback : LBRACKET circuit_expression RBRACKET UNDER LPAREN int RARROW int RPAREN
        """
        
        p[0] = ca.Feedback(p[2], p[6], p[8])
        
    def p_circuit_feedback_2(self, p):
        """
        circuit_feedback : FB LPAREN circuit_expression RPAREN
                     | FB LPAREN circuit_expression COMMA int COMMA int RPAREN
        """
        if len(p) == 9:
            p[0] = ca.Feedback(p[3], p[5], p[7])
        else:
            p[0] = ca.Feedback(p[3])
        
    def p_circuit_permutation(self, p):
        """
        circuit_permutation : PSIGMA LPAREN int_list RPAREN
        """
        p[0] = ca.CPermutation(p[3])
        
    
    def p_int_list(self, p):
        """
        int_list : int_list COMMA int
                 | int
        """
        if len(p) == 2:
            p[0] = (p[1],)
        else:
            p[0] = p[1] + (p[3],)
    
    def p_int(self, p):
        """
        int : ICONST
        """
        p[0] = int(p[1])


    precedence = (
        ('left', 'PLUS'),
        ('left', 'LSERIES'),
        )
        
    def p_error(self, p):
        print self.lexer
        print p, self.lexer.lineno  
        raise Exception