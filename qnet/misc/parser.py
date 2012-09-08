from ply import lex, yacc


import os


class ParsingError(SyntaxError):
    pass


class Parser(object):
    """
    Base class for a lexer/parser that has the _rules defined as methods
    """
    tokens = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.names = {}
        try:
            modname = os.path.splitext(__file__)[0] + "_" + self.__class__.__name__
        except:
            modname = "parser" + "_" + self.__class__.__name__
        self.debugfile = modname + ".dbg"
        self.tabmodule = modname + "_" + "parsetab"
        # print self.debugfile, self.tabmodule

        # Build the lexer and parser

        self.lexer = lex.lex(module=self, debug=self.debug)
        self.parser = yacc.yacc(module=self,
            debug=self.debug,
            debugfile=self.debugfile,
            tabmodule=self.tabmodule)

    def parse(self, inputstring):
        return self.parser.parse(inputstring, lexer=self.lexer)

    def parse_file(self, filename):
        with open(filename, 'r') as inputfile:
            filetext = inputfile.read()
        return self.parse(filetext)