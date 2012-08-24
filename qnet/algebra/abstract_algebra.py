#!/usr/bin/env python
# encoding: utf-8
"""
abstract_algebra.py

Created by Nikolas Tezak on 2011-02-17.
Copyright (c) 2011 . All rights reserved.
"""

from __future__ import division
from itertools import izip
from sympy.core.basic import Basic as SympyBasic
from sympy.core.sympify import sympify, SympifyError
from sympy.printing.latex import latex as sympy_latex
from numpy import ndarray
from functools import reduce as freduce



#def n(obj):
#    try:
#        # print obj, obj.n()
#        return obj.n()
#    except AttributeError:
#        return obj
#
#def is_number(obj):
#    "Check for Python number type"
#    return isinstance(obj, (int, long, float, complex))
#
#
#def is_integral(obj):
#    "Check for Python integral number type"
#    return isinstance(obj, (int, long))
#
#def set_union(*sets):
#    "Return the union of an arbitrary number of sets."
#    return freduce(lambda set1, set2: set1 | set2, sets, set(()))
#
#def set_intersection(*sets):
#    "Return the intersection of an arbitrary number of sets"
#    return freduce(lambda set1, set2: set1 & set2, sets, set(()))
#
#def simplify(obj):
#
#    try:
#        return obj.simplify(**rules)
#    except AttributeError:
#        return obj
#
#
#def NotImplementedFn(*args):
#    """Always return NotImplemented"""
#    return NotImplemented

# def nprint(obj, tab_level= 0):
#     if hasattr(obj, 'nprint'):
#         return obj.nprint(tab_level)
#     return str(obj)


#define our own errors
class AlgebraError(Exception):
    """
    Base class for all errors concerning the mathematical definitions and rules of an algebra.
    """
    pass
    
class CannotSimplify(AlgebraError):
    """
    Raised when an expression cannot be further simplified
    """
    pass

class WrongSignature(AlgebraError):
    """
    Raise when an operation is instantiated with operands of the wrong signature.
    """
    pass


#class OperandsError(AlgebraError):
#    """
#    Raised when the operands of an operation violate a specific rule.
#    """
#    def __init__(self, msg, *operands):
#        AlgebraError.__init__(self, msg + ": " + str(map(repr, operands)))
#
#class CannotCompare(AlgebraError):
#    """
#    Raised when comparison between two objects is not implemented by a certain function.
#    """
#    pass
#
#class SymbolIdentifierTaken(AlgebraError):
#    """
#    Raised on an attempt to create a new symbol with an existing identifier.
#    """
#    def __init__(self, existing_symbol, existing_args, new_args):
#        AlgebraError.__init__(self, "A symbol with identifier %s was already defined (%s).\n Newly requested args: %r"
#                                        % (existing_symbol.identifier, existing_symbol, new_args))
#
#
#
class Expression(object):
    """Abstract base class defining the basic methods any Expression object should implement."""


    def substitute(self, var_map):
        """
        Substitute symbols for other expressions via
        var_map = {Symbol_1_object:SubstituteExpression_1,... }
        and return self with all substitutions carried out.
        """
        if self in var_map:
            return var_map[self]

        return self

    def tex(self):
        """
        Return a string containing a TeX-representation of self.
        """
        return str(self)

    def _latex_(self):
        # print "called _latex_ on %s" % self
        return self.tex()

    def _repr_latex_(self):
        return "$%s$" % self.tex()

    def free_of(self, *symbols):
        """
        Check if expression is independent of all passed symbols
        """
        return len(set(symbols) & self.symbols()) == 0

    def symbols(self):
        """
        Return set of symbols contained within the expression.
        """
        raise NotImplementedError(self.__class__.__name__)


    def __hash__(self):
        """
        Provide a hashing mechanism for self.
        """
        raise NotImplementedError(self.__class__.__name__)
#
    def __eq__(self, other):
        """
        Implements a very strict definition of 'self == other'.
        This should be overloaded where appropriate.
        """
        if self is other:
            return True
        return False

    def __ne__(self, other):
        """
        If it is well-defined (i.e. boolean), simply return
        the negation of 'self.__eq__(other)'
        Otherwise return NotImplemented.
        """
        eq = self.__eq__(other)
        if type(eq) is bool:
            return not eq
        return NotImplemented

    def simplify(self):
        return self

#
#    def __cmp__(self, other):
#        """
#        Return
#            -1 if self < other,
#            0 if self == other,
#            +1 if other > self.
#
#        A strict and well-defined (i.e. unique) order relation on all expressions is important
#        for any kind of implementation of a 'canonical form' of an expression.
#        """
#        if self is other:
#            return 0
#
#        s_alg = algebra(self)
#        o_alg = algebra(other)
#
#        # if they have the same algebra
#        if s_alg is o_alg:
#            # compare within the algebra
#            return s_alg.cmp_within_algebra(self, other)
#
#        # otherwise compare their algebras
#        return cmp(s_alg, o_alg)
#
#    def simplify(self):
#        raise NotImplementedError(self.__class__.__name__)
#
    def mathematica(self):
        raise NotImplementedError(self.__class__.__name__)
#
# def expand(expr):
#     """
#     (Safe) expand: Distributively expand out all products between sums.
#     """
#     try:
#         return expr.expand()
#     except AttributeError:
#         return expr
#
#
def substitute(expr, var_map):
    """
    (Safe) substitute: Substitute objects for symbols
    """
    try:
        return expr.substitute(var_map)
    except AttributeError:
        try:
            # try substituting sympy_objects
            var_map = dict((k,v) for (k,v) in var_map.items() if not isinstance(k, Expression) and not isinstance(v, Expression))
            return expr.subs(var_map)
        except AttributeError:
            return expr
        except Exception, e:
            print "Warning while trying to substitute in %s an error occured: %s" % (expr, e)
            return expr
#
#def evalf(expr):
#    """
#    (Safe) evalf: Evaluate all (sub-) expressions that can be converted to pure numerical form.
#    """
#    try:
#        return expr.evalf()
#    except AttributeError:
#        return expr
#
#
def tex(obj):
    """
    Return a LaTeX string-representation of obj.
    """
    if isinstance(obj, str):
        return identifier_to_tex(obj)
    if is_number(obj):
        return format_number_for_tex(obj)
    if isinstance(obj, SympyBasic):
        return sympy_latex(obj)#[1:-1] #trim '$' at beginning and end of returned string
    try:
        return obj.tex()
    except AttributeError:
        try:
            return obj._latex_()
        except AttributeError:
            return str(obj)
#
#def mathematica(obj):
#    """
#    Return a Mathematica string-representation of obj
#    """
#    if isinstance(obj, str):
#        return identifier_to_mathematica(obj)
#    if is_number(obj):
#        return format_number_for_mathematica(obj)
#    if isinstance(obj, SympyBasic):
#        return capitalize_sympy_functions_for_mathematica(
#                        re.compile(r'([A-Za-z0-9]+)\(([^\)]+)\)').sub(
#                                        r"\1[\2]", identifier_to_mathematica(str(obj)).replace("**","^")))
#    try:
#        return obj.mathematica()
#    except AttributeError:
#        return str(obj)
#
#def capitalize_sympy_functions_for_mathematica(string):
#    words = ("cos", "sin", "exp", "sqrt", "conjugate", "cosh", "sinh")
#    return reduce(lambda a, b: a.replace(b, b[0].upper() + b[1:]), words, string)
#
#
## def diff(expr, wrt, n = 1):
##     """
##     Safe differentiation.
##     """
##     if n < 0:
##         raise AlgebraError()
##     if n == 0:
##         return expr
##     try:
##         return diff(expr.diff(wrt), wrt, n - 1)
##     except AttributeError:
##         return 0
#
#def free_of(expr, *symbols):
#    """
#    Safe free_of
#    """
#    try:
#        return expr.free_of(*symbols)
#    except AttributeError:
#        return True
#
#def symbols(expr):
#    """
#    Safe symbols
#    """
#    try:
#        return expr.symbols()
#    except AttributeError:
#        try:
#            return expr.atoms()
#        except AttributeError:
#            return set(())
#
#
#def format_number_for_tex(num):
#    if num == 0: #also True for 0., 0j
#        return "0"
#    if isinstance(num, complex):
#        if num.real == 0:
#            if num.imag == 1:
#                return "i"
#            if num.imag == -1:
#                return "(-i)"
#            if num.imag < 0:
#                return "(-%si)" % format_number_for_tex(-num.imag)
#            return "%si" % format_number_for_tex(num.imag)
#        if num.imag == 0:
#            return format_number_for_tex(num.real)
#        return "(%s + %si)" % (format_number_for_tex(num.real), format_number_for_tex(num.imag))
#    if num < 0:
#        return "(%g)" % num
#    return "%g" % num
#
#def format_number_for_mathematica(num):
#    if num == 0: #also True for 0., 0j
#        return "0"
#    if isinstance(num, complex):
#        if num.imag == 0:
#            return format_number_for_tex(num.real)
#        return "Complex[%g,%g]" % (num.real, num.imag)
#
#    return "%g" % num
#
#
#
#
#greek_letter_strings = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", \
#                        "zeta", "eta", "theta", "vartheta", "iota", "kappa", \
#                        "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", \
#                        "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", \
#                        "varphi", "chi", "psi", "omega", \
#                        "Gamma", "Delta", "Theta", "Lambda", "Xi", \
#                        "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]
#greekToLatex = {"alpha":"Alpha", "beta":"Beta", "gamma":"Gamma", "delta":"Delta", "epsilon":"Epsilon", "varepsilon":"Epsilon", \
#                        "zeta":"Zeta", "eta":"Eta", "theta":"Theta", "vartheta":"Theta", "iota":"Iota", "kappa":"Kappa", \
#                        "lambda":"Lambda", "mu":"Mu", "nu":"Nu", "xi":"Xi", "pi":"Pi", "varpi":"Pi", "rho":"Rho", \
#                        "varrho":"Rho", "sigma":"Sigma", "varsigma":"Sigma", "tau":"Tau", "upsilon":"Upsilon", "phi": "Phi", \
#                        "varphi":"Phi", "chi":"Chi", "psi":"Psi", "omega":"Omega", \
#                        "Gamma":"CapitalGamma", "Delta":"CapitalDelta", "Theta":"CapitalTheta", "Lambda":"CapitalLambda", "Xi":"CapitalXi", \
#                        "Pi":"CapitalPi", "Sigma":"CapitalSigma", "Upsilon":"CapitalUpsilon", "Phi":"CapitalPhi", "Psi":"CapitalPsi", "Omega":"CapitalOmega"
#                }
#
#import re
#def identifier_to_tex(identifier):
#    """
#    If an identifier contains a greek symbol name as a separate word,
#    (e.g. 'my_alpha_1' contains 'alpha' as a separate word, but 'alphaman' doesn't)
#    add a backslash in front.
#    """
#    identifier = reduce(lambda a,b: "{%s_%s}" % (b, a), ["{%s}" % part for part in reversed(identifier.split("__"))])
#    p = re.compile(r'([^\\A-Za-z]?)(%s)\b' % "|".join(greek_letter_strings))
#    return p.sub(r'\1{\\\2}', identifier)
#
#
#
#
#
#def identifier_to_mathematica(identifier):
#    """
#    If an identifier contains a greek symbol name as a separate word,
#    (e.g. 'my_alpha_1' contains 'alpha' as a separate word, but 'alphaman' doesn't)
#    add a backslash in front.
#    """
#    identifier = reduce(lambda a,b: "Subscript[%s,%s]" % (b, a), reversed(identifier.split("__")))
#    p = re.compile(r'\b(%s)\b' % "|".join(greek_letter_strings))
#    repl = lambda m:  r"\[" + greekToLatex[m.group(1)] + "]"
#    return p.sub(repl, identifier)
#
#
#
#
#class Symbol(Expression):
#    """
#    Abstract base class for Symbolic objects.
#    All Symbols (across all algebras) are uniquely identified by an identifier string.
#
#    Once a symbol of a given identifier has been created, an additional attempt
#    to create a symbol of the same identifier will either
#        - yield the same object, only if it is being instantiated with exactly the same additional arguments
#        - lead to a 'SymbolIdentifierTaken' exception.
#    """
#
#    symbol_cache = {}
#
#    def __new__(cls, identifier, *args):
#        # perform cache lookup
#        # if not hasattr(cls, 'symbol_cache'):
#        #     cls.symbol_cache = {}
#        instance, i_args = cls.symbol_cache.get(identifier, (False, None))
#        if instance:
#            if args != i_args:
#                raise SymbolIdentifierTaken(instance, i_args, args)
#
#            return instance
#        instance = Expression.__new__(cls)
#        cls.symbol_cache[identifier] = instance, args
#        return instance
#
#    def __init__(self, identifier, *args):
#        self._identifier = identifier
#        self._hash = hash((self.__class__, identifier, args))
#        self._args = args
#
#    def __hash__(self):
#        return self._hash
#
#    def __eq__(self, other):
#        """
#        For any identifier (and other parametric args)
#        there exists only a single instance,
#        equality implies actual instance equality
#        """
#        return self is other
#
#    # def expand(self):
#    #     return self
#
#    def substitute(self, var_map):
#        if self in var_map:
#            return var_map[self]
#        if self.identifier in var_map:
#            return var_map[self.identifier]
#        return self
#
#    def __repr__(self):
#        return "%s%r" % (self.__class__.__name__, (self._identifier,) + self._args)
#
#    def evalf(self):
#        return self
#
#    def simplify(self):
#        return self
#
#    def tex(self):
#        return identifier_to_tex(self._identifier)
#
#    def free_of(self, *symbols):
#        return not self in symbols and not self.identifier in symbols
#
#    def symbols(self):
#        return set([self])
#
#    @property
#    def identifier(self):
#        return self._identifier
#
#    def mathematica(self):
#        return self.identifier
#
#
#CHECK_OPERANDS = False
#
#
#class insert_default_rules(object):
#
#    def __init__(self, default_rules):
#        self.default_rules = default_rules
#
#    def __call__(self, function_to_decorate):
#        # print function_to_decorate
#        def function_with_default_rules(*args, **additional_rules):
#            # print args, additional_rules
#            rules = self.default_rules.copy()
#            rules.update(additional_rules)
#            return function_to_decorate(*args)
#
#        return function_with_default_rules
#



class Operation(Expression):
    """
    Abstract base class for all operations,
    where the operands themselves are also expressions.
    """
#    @classmethod
#    def check_operands(cls, *operands):
#        raise NotImplementedError(cls.__name__)
#
    _hash = None

    ##### instance methods/properties
    def __init__(self, *operands):
        """
        Create a symbolic operation with the operands. No filtering/simplification
        of operands is performed at this stage, but if the global var CHECK_OPERANDS
        is True, the operands are checked according to the rules.
        """
#        if CHECK_OPERANDS:
#            self.__class__.check_operands(*operands)
            
        self._operands = operands
        
#        if len(operands) == 0:
#            raise Exception('Need at least one operand')
#
#        self._hash = None   #lazy hash creation, i.e. when it is actually needed
    
    @property
    def operands(self):
        return self._operands
        
    
    def __hash__(self):
        if not self._hash:
            self._hash = hash((self.__class__, self.operands))
        return self._hash


    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(map(repr, self.operands)))
    
#    def mathematica(self):
#        return "%s[%s]" % (self.__class__.__name__, ", ".join(map(mathematica, self.operands)))

    # def expand(self):
    #     return self.__class__.apply_with_rules(*map(expand, self.operands))
    
    def substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        return self.__class__.create(*map(lambda o: substitute(o, var_map), self.operands))
    
#    def evalf(self):
#        return self.__class__.create(*map(evalf, self.operands))
    
    def __eq__(self, other):
        return type(self) == type(other) and self.operands == other.operands
        
    def symbols(self):
        return set_union(*[symbols(op) for op in self.operands])
        
    def free_of(self, *symbols):
        return all((free_of(op, *symbols) for op in self.operands))
    
#    def simplify(self):
#        """
#        Simplify the operation according to some rules by reapplying it
#        (with these rules) to the simplified operands.
#        """
#        s_ops = []
#
#        for o in self.operands:
#            try:
#                s_ops.append(simplify(o))
#
#            except CannotSimplify:
#                s_ops.append(o)
#
#        return self.__class__.create(*s_ops)
#
        
    
    
    @classmethod
    def create(cls, *operands):
        """
        Instead of directly instantiating an operation object, 
        it should always be instantiated by applying the operations with
        all (default) rules. This ensures that no invalid expressions are created.
        """
        return cls(*operands)

    @classmethod
    def order_key(cls, a):
        if hasattr(a, "_key_"):
            return a._key_()
        return a


    def _key_(self):
        return (self.__class__.__name__,) + self.operands

        
             

##PRINT_PRETTY = False
#
#class MultiaryOperation(Operation):
#    """Base class for all operations that require two or more operands."""
#
#    ## TODO clean up operation/filter_operands call interface
#
#
#
#
#    operation_symbol = ' # ' #overwrite this!!!
#
#    tex_symbol = ' \diamond '
#
#    default_rules = {}
#
#
#    def __str__(self):
#        return "(%s)" % self.operation_symbol.join(map(str, self.operands))
#
#    def tex(self):
#        return "\left( %s \\right)" % self.tex_symbol.join(map(tex, self.operands))
#
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def check_operands(cls, *operands):
#        if len(operands) < 2:
#            raise OperandsError('Need at least two operands', *operands)
#
#    @classmethod
#    def filter_operands(cls, operands):
#        return operands
#
#    @classmethod
#    def handle_zero_operands(cls):
#        raise NotImplementedError("Not implemented yet for class %s" % cls.__name__)
#
#    @classmethod
#    def handle_single_operand(cls, operand):
#        return operand
#
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def create(cls, *operands):
#
#        operands = cls.filter_operands(operands)
#
#        if len(operands) == 0:
#            return cls.handle_zero_operands()
#
#        elif len(operands) == 1:
#            return cls.handle_single_operand(operands[0])
#
#        return cls(*operands)
#
#
#class AssociativeOperation(MultiaryOperation):
#
#    #RULES inherited from MultiaryOperation
#    default_rules = MultiaryOperation.default_rules.copy()
#
#    #plus new rule
#    default_rules.update(expand_associatively = True)
#
#
#    def __str__(self):
#        if PRINT_PRETTY:
#            return "\n(%s\n)" % ("\n    " + self.operation_symbol).join([str(op).replace("\n", "\n    ") for op in self.operands])
#        return "(%s)" % (self.operation_symbol).join([str(op)for op in self.operands])
#
#    def __repr__(self):
#        if PRINT_PRETTY:
#            return "%s(\n    %s\n)" % (self.__class__.__name__, ",\n    ".join([repr(op).replace("\n", "\n    ") for op in self.operands]))
#        return "%s(%s)" % (self.__class__.__name__, ",".join([repr(op) for op in self.operands]))
#
#    @classmethod
#    def check_associatively_expanded(cls, *operands):
#        if any((isinstance(op, cls) for op in operands)):
#            raise OperandsError("Not expanded", *operands)
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def check_operands(cls, *operands):
#        MultiaryOperation.check_operands.im_func(cls, *operands)
#
#        if rules.get('expand_associatively', cls.default_rules.get('expand_associatively')):
#            cls.check_associatively_expanded(*operands)
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def create(cls, *operands):
#
#        expand_associatively = rules['expand_associatively']
#
#        if expand_associatively:
#            operands = cls.expand_associatively(operands)
#
#        operands = cls.filter_operands(operands)
#        operands = cls.combine_binary(*operands)
#
#        # if expand_associatively:
#        #     operands = cls.expand_associatively(operands)
#
#
#        if len(operands) == 0:
#            return cls.handle_zero_operands()
#        elif len(operands) == 1:
#            return cls.handle_single_operand(operands[0])
#
#        return cls(*operands)
#
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def combine_binary(cls, *operands):
#        for i, (lhs, rhs) in enumerate(izip(operands[:-1], operands[1:])):
#            try:
#                combined = cls.simplify_binary(lhs, rhs)
#
#                # allow for the possibility that the simplified expression
#                # is still a composite object of the type cls, be careful to raise the CannotSimplify exception,
#                # i.e., within simplify_binary NEVER return cls(lhs,rhs) as this would lead to an infinite loop!!!
#
#                combined_operands = (combined,) if not isinstance(combined, cls) else combined.operands
#
#                if i > 0: #maybe the combined object can now be combined with the previous operand
#                    return operands[ : i-1] +  cls.combine_binary(operands[i-1], *(combined_operands + operands[i + 2 : ]))
#
#                return operands[ : i] +  cls.combine_binary(*(combined_operands + operands[i + 2 : ]))
#            except CannotSimplify:
#                pass
#        return operands
#
#    @classmethod
#    def simplify_binary(cls, lhs, rhs):
#        if not hasattr(cls, 'has_been_warned'):
#            print "WARNING: please implement this correctly for class %s" % cls.__name__
#            cls.has_been_warned = True
#        raise CannotSimplify
#
#    @classmethod
#    def expand_associatively(cls, operands):
#        operands = operands
#        for i, o in enumerate(operands):
#            if isinstance(o, cls):
#                return operands[ : i] + o.operands + cls.expand_associatively(operands[i + 1 : ])
#        return operands
#
#
#def expand(obj):
#    try:
#        return obj.expand()
#    except AttributeError:
#        return obj
#
#
#
#
#
#class Addition(AssociativeOperation):
#
#    operation_symbol = ' + '
#    tex_symbol = operation_symbol
#
#    def evalf(self):
#        return reduce(lambda a, b: a + b, map(evalf, self.operands))
#
#    @classmethod
#    def filter_operands(cls, operands):
#        operands = filter(lambda o: o != 0, operands)
#        return sorted(operands)
#
#    @classmethod
#    def handle_zero_operands(cls):
#        return 0
#
#    # def diff(self, wrt):
#    #     return sum([diff(op, wrt) for op in self.operands])
#
#    def expand(self):
#        return self.create(*map(expand, self.operands))
#
#class Multiplication(AssociativeOperation):
#    operation_symbol = ' * '
#    tex_symbol = ' '
#
#    @classmethod
#    def filter_operands(cls, operands):
#        return filter(lambda op: op != 1, operands)
#
#    @classmethod
#    def handle_zero_operands(cls):
#        return 1
#
#
#    def expand(self):
#        operands = map(expand, self._operands)
#        for i, o in enumerate(operands):
#            if isinstance(o, Addition):
#                pre_factor = operands[:i]
#                post_factor = operands[i+1:]
#                new_summands = map(lambda f: self.__class__.create(*(pre_factor + [f] + post_factor)), o._operands)
#                return expand(o.__class__.create(*new_summands))
#        return self.__class__.create(*operands)
#
#    # def diff(self, wrt):
#    #     result = 0
#    #     for i, op in self.operands:
#    #         result += (self.__class__.apply_with_rules(*(self.operands[i:] + (diff(op, wrt),) + operands[i+1:])))
#    #     return result
#
#    def evalf(self):
#        return reduce(lambda a, b: a * b, map(evalf, self.operands))
#
#
#
#class BinaryOperation(MultiaryOperation):
#
#    @property
#    def lhs(self):
#        return self._operands[0]
#
#    @property
#    def rhs(self):
#        return self._operands[1]
#
#    def __init__(self, lhs, rhs):
#        Operation.__init__(self, lhs, rhs)
#
#    @classmethod
#    def create(cls, lhs, rhs):
#        return MultiaryOperation.create.im_func(cls, lhs, rhs)
#
#
#class Power(BinaryOperation):
#    default_rules = {'combine_power_powers': True,         # (a**b)**c = a**(b*c)
#                    'expand_fraction_powers': True,        # (a/b)**c = a**c/b**c
#                    }
#
#    operation_symbol = '**'
#
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def check_operands(cls, base, exponent):
#        if base == 1 or base == 0 or exponent == 1 or exponent == 1:
#            raise OperandsError('', base, exponent)
#
#    @classmethod
#    @insert_default_rules(default_rules)
#    def create(cls, base, exponent):
#
#        if is_number(base) and is_number(exponent):
#            return base ** exponent
#
#        if base == 1:
#            return 1
#
#        if exponent == 0:
#            return 1
#
#        if exponent == 1:
#            return base
#
#
#        if isinstance(base, cls):
#            return cls(base.base, base.exponent * exponent)
#
#        if isinstance(base, CoefficientTermProduct):
#            return cls.create(base.coeff, exponent) * cls.create(base.term, exponent)
#
#        return cls(base, exponent)
#
#    def tex(self):
#        return "{%s}^{%s}" % (tex(self.base), tex(self.exponent))
#
#    def __str__(self):
#        return "%s**%s" % (self.base, self.exponent)
#
#    @property
#    def base(self):
#        return self._operands[0]
#
#    @property
#    def exponent(self):
#        return self._operands[1]
#
#
#    # def diff(self, wrt):
#    #     """The exponent is always assumed to be a constant integer, so it is not differentiated."""
#    #     base, exponent = self.operands
#    #
#    #     d_b = diff(base, wrt)
#    #
#    #     return exponent * base**(exponent - 1) * d_b
#
#
#    def evalf(self):
#        return evalf(self.base)**evalf(self.exponent)
#
#
#class CoefficientTermProduct(BinaryOperation):
#    """
#    Abstract class to implement a cross-algebra product: coeff * term
#    """
#
#    operation_symbol = ' * '
#    tex_symbol = " "
#
#
#    @property
#    def coeff(self):
#        return self._operands[0]
#
#    @property
#    def term(self):
#        return self._operands[1]
#
#    def expand(self):
#        # this only expands the term, not the coefficient
#        coeff, term = expand(self.coeff), expand(self.term)
#
#        if isinstance(term, Addition): #only expand for term
#            return term.__class__.create(*(map( lambda t: self.__class__.create(coeff, t), term.operands)))
#        return self.__class__.create(coeff, term)
#
#    # def diff(self, wrt):
#    #     return diff(self.coeff, wrt) * self.term + self.coeff * diff(self.term, wrt)
#
#
#
#
## class Fraction(BinaryOperation):
##     """
##     Abstract superclass for fractions:
##     numerator/denominator == Fraction(numerator, denominator).
##     """
##
##     operation_symbol = " / "
##
##     @property
##     def numerator(self): #zaehler
##         return self._operands[0]
##
##     @property
##     def denominator(self):#nenner
##         return self._operands[1]
##
##     def tex(self):
##         return "\\frac{%s}{%s}" % (tex(self.numerator), tex(self.denominator))
##
##
##     def diff(self, wrt):
##         return self.__class__.apply_with_rules(diff(self.numerator, wrt) * self.denominator - self.numerator * diff(self.denominator), self.denominator**2)
##
##     def expand(self):
##         return self.__class__.apply_with_rules(expand(self.numerator), self.denominator)
#
#class UnaryOperation(Operation):
#    """
#    Abstract superclass for all unary operations,
#    i.e. operations with a single operand.
#    """
#
#    @classmethod
#    def check_operands(cls, *operands):
#        if len(operands) != 1:
#            raise OperandsError('Wrong number of operands', *operands)
#
#    def __init__(self, operand):
#        Operation.__init__(self, operand)
#
#    @property
#    def operand(self):
#        return self.operands[0]
#
#
#class Number:
#    """Dummy class to assign an algebra to Python built-in numbers."""
#    pass
#
#
#
#
#
#def algebra(obj):
#    """
#    Return the associated algebra of obj.
#
#    obj may be:
#        1) a python number:
#            algebra(5) == algebra(1.) == algebra(1+0j) == Number
#        2) an instance of a subclass of Algebra:
#            algebra(ScalarSymbol) == Scalar
#        3) a subclass of Algebra:
#            algebra(ScalarSymbol) == algebra(Scalar) == Scalar
#    """
#    if isinstance(obj, Algebra) or isinstance(obj, AlgebraType):
#        return obj.algebra
#    if isinstance(obj, SympyBasic):
#        # print "sympyBasic:", obj
#        return SympyBasic
#    if is_number(obj):
#        return Number
#    if isinstance(obj, ndarray):
#        return None
#    try:
#        sympify(obj)
#        # print 'sympifiable:', obj
#        return Number #enforce sympification
#    except SympifyError:
#        pass
#
#    return None
#
#
#class AlgebraType(type):
#    """
#    This serves as a Metaclass for Algebra classes.
#    See http://en.wikibooks.org/wiki/Python_Programming/MetaClasses
#    for some information what metaclasses actually do.
#    In our case, we would like to use it only to achieve that any direct Algebra
#    subclass (e.g. Scalar) gets a class property called 'algebra' referencing itself.
#    Subclasses of the Algebra subclasses (e.g. ScalarAddition) then simply inherit this
#    class property.
#    Example
#
#        class MyAlgebra(Algebra):
#            pass
#
#        class MyAlgebraSubclass(MyAlgebra):
#            pass
#
#        MyAlgebra.algebra == MyAlgebra            # True, because MyAlgebra is direct subclass of Algebra
#        MyAlgebraSubclass.algebra == MyAlgebra    # True, because this class attribute is inherited from MyAlgebra
#
#    """
#    def __new__(cls, clsname, clssuper, clsdict):
#        """
#        Called when a new class is defined with the name clsname,
#        the superclasses clssuper, and the properties and methods defined in clsdict.
#        cls points to the metaclass, i.e. to AlgebraType (or a submetaclass of this).
#        """
#
#        #create the actual class
#        cls = type.__new__(cls, clsname, clssuper, clsdict)
#
#        # If we're creating the Algebra class itself, then skip ahead.
#        if clsname != 'Algebra':
#
#            # if the new class is a direct subclass of Algebra
#            if Algebra in clssuper:
#
#                #store a self reference of the class in the 'private' property _algebra
#                cls._algebra = cls
#
#            else:
#                #otherwise make sure that it is derived off some other Algebra
#                if not any((issubclass(sup, Algebra) for sup in clssuper)):
#                    print clsname, clssuper
#                    raise Exception()
#
#        return cls
#
#
#    # make a read-only public property for the specific algebra class itself
#    @property
#    def algebra(self):
#        return self._algebra
#
#
#def make_binary_operation_method(operation_map_name, operation_string = ''):
#    """
#    Return a method for a binary operation between to algebraic expressions.
#    """
#
#    def operation(a, b):
#        operation_map = getattr(algebra(a), operation_map_name, {})
#        # print operation_map_name[:-4], a,b
#        return operation_map.get(algebra(b), NotImplementedFn)(a, b)
#
#    if operation_string != '':
#        operation.__doc__ = """
#        To implement the operation '%s'
#        for the specific algebras of 'self' and of 'other',
#        implement the operation in a function
#            >> def operation(self, other):
#            >>     # ...
#        and add an entry to the appropriate operation_map:
#            >> self.algebra.%s[algebra(other)] = operation
#        """ % (operation_string, operation_map_name)
#    return operation
#
#
###### TODO: QUICK FIX!
#old_cmp = cmp
#def cmp(a, b):
#    try:
#        return old_cmp(a,b)
#    except TypeError:
#        if isinstance(a, complex) or isinstance(b, complex):
#            return old_cmp(a.real, b.real) or old_cmp(a.imag, b.imag)
#        raise TypeError('Woops')
#
#
#
#class Algebra(object):
#    """
#    Basis class from which all Algebras should directly be derived.
#    Every direct subclass of Algebra has an attribute 'algebra' that points to itself.
#    For more information on this see the documentation for AlgebraType.
#
#    To mathematical operations for all MyAlgebra objects can be changed at runtime
#    by changing the according map property.
#    To implement the Addition of two MyAlgebra objects, this code would work:
#
#        def add_my_algebra_objects(a,b):
#            # do and return something ...
#
#        MyAlgebra.add_map[MyAlgebra] = add_my_algebra_objects
#
#    and if we now also wanted to implement addition with another algebra such
#    as python numbers, we need to specify both addition from left and from right:
#
#        def add_my_algebra_object_to_number(obj, number):
#            # do and return something
#
#        # notice the order of the arguments!
#        # this is consistent with how one would implement
#        # the __radd__ method.
#        def add_number_to_my_algebra_object(obj, number):
#            # do and return something (else?)
#
#        MyAlgebra.add_map[Number] = add_my_algebra_object_to_number
#        MyAlgebra.radd_map[Number] = add_number_to_my_algebra_object
#
#
#    This kind of mechanism is implemented
#    for all of the binary operators listed in the first column.
#        +   add_map     radd_map
#        -   sub_map     rsub_map
#        /   div_map     rdiv_map
#        *   mul_map     rmul_map
#        **  pow_map     rpow_map
#        <<  lshift_map  rlshift_map
#        >>  rshift_map  rrshift_map
#
#    The second and third column specify the names which the operation dictionary
#    should have as algebra class properties for operating from left and from the right.
#    Once you have specified the mathematical operation between MyAlgebra via
#    the operation maps of MyAlgebra, you need not re-implement this behaviour
#    within the other Algebra class.
#    This way, only one Algebra has to 'know about' the other.
#    E.g. Operators should know Scalars, but not vice versa.
#    """
#
#
#    # specify that the type of the Algebra class is actually the AlgebraType
#    __metaclass__ = AlgebraType
#
#    @property
#    def algebra(self):
#        """
#        In contrast to the property of the class itself,
#        this is a property of each instance of an algebra object.
#        """
#        return self.__class__._algebra
#
#
#
#    __mul__ = make_binary_operation_method('mul_map', 'self * other')
#    __rmul__ = make_binary_operation_method('rmul_map', 'other * self')
#
#    __add__ = make_binary_operation_method('add_map', 'self + other')
#    __radd__ = make_binary_operation_method('radd_map', 'other + self')
#
#    __sub__ = make_binary_operation_method('sub_map', 'self - other')
#    __rsub__ = make_binary_operation_method('rsub_map', 'other - self')
#
#
#
#    __div__ = make_binary_operation_method('div_map', 'self / other')
#    __rdiv__ = make_binary_operation_method('rdiv_map', 'other / self')
#
#    __truediv__ = __div__
#    __rtruediv__ = __rdiv__
#
#    __pow__ = make_binary_operation_method('pow_map', 'self**other')
#    __rpow__ = make_binary_operation_method('rpow_map', 'other**self')
#
#    __lshift__ = make_binary_operation_method('lshift_map', 'self << other')
#    __rlshift__ = make_binary_operation_method('rlshift_map', 'other << self')
#
#    __rshift__ = make_binary_operation_method('rshift_map', 'self >> other')
#    __rrshift__ = make_binary_operation_method('rrshift_map', 'other >> self')
#
#    def cmp_within_algebra(self, other):
#        """
#        General comparison function between objects of the same algebra.
#        See also the documentation of Expression.__cmp__
#
#
#        Algebra Subclasses that feature expression that aren't subclasses of
#        Symbol, CoefficientTermProduct or Operation, need to overload this method.
#        If the overloaded implementation calls this method, it should make sure,
#        that any additional types of expressions must come AFTER all of the
#        above-mentioned ones.
#        """
#
#        """
#        for any type of coefficientTermProduct:
#        compare terms first and only if they are equal, compare coefficients
#        """
#        if isinstance(self, CoefficientTermProduct):
#            if isinstance(other, CoefficientTermProduct):
#                return cmp(self.algebra, other.algebra) \
#                        or cmp(self.term, other.term) \
#                        or cmp(self.coeff, other.coeff)
#            return cmp(self.term, other) or cmp(self.coeff, 1)
#        if isinstance(other, CoefficientTermProduct):
#            return cmp(self, other.term) or cmp(1, other.coeff)
#
#        """
#        For any type of power, compare bases first and then exponents
#        """
#        if isinstance(self, Power):
#            if isinstance(other, Power):
#                return cmp(self.algebra, other.algebra) \
#                    or cmp(self.base, other.base) \
#                    or cmp(self.exponent, other.exponent)
#            return cmp(self.base, other)
#        if isinstance(other, Power):
#            return cmp(self, other.base)
#
#
#
#        """
#        Among symbols, compare the class and then the identifier.
#        Individual symbols should come before anything else.
#        """
#        if isinstance(self, Symbol):
#            if isinstance(other, Symbol):
#                return  cmp(self.__class__.__name__, other.__class__.__name__) \
#                        or cmp(self.identifier, other.identifier)
#            return -1
#        if isinstance(other, Symbol):
#            return +1
#
#        """
#        Among operations, compare the class and then the operands.
#        """
#        if isinstance(self, Operation):
#            if isinstance(other, Operation):
#                return cmp(self.__class__.__name__, other.__class__.__name__) \
#                        or cmp(self.operands, other.operands)
#            return -1
#
#        if isinstance(other, Operation):
#            return +1
#
#
#        raise CannotCompare(repr((self, other)))
#

inf = float('inf')

def match_range(pattern):
    if isinstance(pattern, Wildcard):
        if pattern.mode == Wildcard.single:
            return 1,1
        if pattern.mode == Wildcard.one_or_more:
            return 1, inf
        if pattern.mode == Wildcard.zero_or_more:
            return 0, inf
        raise ValueError()
    if isinstance(pattern, PatternTuple):
        if len(pattern):
            a0, b0 = match_range(pattern[0])
            a1, b1 = match_range(pattern[1:])
#            a1, b1 = match_range(PatternTuple(pattern[1:]))
            return a0 + a1, b0 + b1
        return 0, 0
    return 1, 1


class OperandsTuple(tuple):
#    def __new__(cls, arg):
#        arg = tuple(arg)
#        if len(arg) == 1:
#            return arg[0]
#        return tuple.__new__(cls, arg)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return OperandsTuple(super(OperandsTuple, self).__getitem__(item))
        return super(OperandsTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return OperandsTuple(super(OperandsTuple, self).__getslice__(i,j))

class PatternTuple(tuple):

#    def __new__(cls, arg):
#        arg = tuple(arg)
#        if len(arg) == 1:
#            return arg[0]
#        return tuple.__new__(cls, arg)


    def __getitem__(self, item):
        if isinstance(item, slice):
            return OperandsTuple(super(PatternTuple, self).__getitem__(item))
        return super(PatternTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return PatternTuple(super(PatternTuple, self).__getslice__(i,j))


class NamedPattern(Operation):
    pass

def trace(fn):
    def tfn(*args, **kwargs):
        print "[", "-"* 40
        ret =  fn(*args, **kwargs)
        print "{}({},{}) called".format(fn.__name__, ", ".join(repr(a) for a in args),
                                        ", ".join(str(k)+"="+repr(v) for k,v in kwargs.items()))
        print "-->", repr(ret)
        print "-"* 40,"]"
        return ret


    return tfn

def flatten(seq):
    sres = []
    for s in seq:
        if isinstance(s, (PatternTuple, OperandsTuple)):
            sres += list(s)
        else:
            sres.append(s)
    return sres

#@trace
def update_pattern(expr, match_obj):

    if isinstance(expr, Wildcard):
        if expr.name in match_obj:
            return match_obj[expr.name]
    elif isinstance(expr, (PatternTuple, OperandsTuple)):
        return expr.__class__(flatten([update_pattern(o, match_obj) for o in expr]))
    elif isinstance(expr, Operation):
        return expr.__class__(*flatten([update_pattern(o, match_obj) for o in expr.operands]))

    return expr



#@trace
def match(pattern, expr):

    if pattern is expr:
        return Match()

    a, b = match_range(pattern)

    if isinstance(expr, OperandsTuple):
        l = len(expr)
    else:
        l = 1

    if not a <= l <= b:
        return False

    if isinstance(pattern, PatternTuple):
        if not len(pattern):
            assert l == 0
            return Match()

        p0 = pattern[0]
        prest = pattern[1:]

        if isinstance(expr, OperandsTuple):

            if isinstance(p0, Wildcard) and p0.mode != Wildcard.single:
                a0, b0 = match_range(p0)
                for k in range(a0, min(l,b0)+1):
                    o0 = expr[:k]
                    orest = expr[k:]
                    m0 = match(p0, o0)
                    if m0:
                        if len(m0):
                            mrest = match(update_pattern(prest, m0), orest)
                        else:
                            mrest = match(prest, orest)
                        if mrest:
                            return m0 + mrest

                return False
            else:
                m0 = match(p0, expr[0])
                if m0:
                    orest = expr[1:]
                    if len(m0):
                        mrest = match(update_pattern(prest, m0), orest)
                    else:
                        mrest = match(prest, orest)
#                    print m0, update_pattern(prest, m0), mrest
                    if mrest:
                        return m0 + mrest
                return False

    elif isinstance(pattern, Wildcard):
        if pattern.mode == Wildcard.single:
            if isinstance(expr, OperandsTuple):
                assert len(expr) == 1
                expr = expr[0]
            if pattern.head and not isinstance(expr, pattern.head):
                return False
            if pattern.condition and not pattern.condition(expr):
                return False
            if pattern.name:
                return Match({pattern.name: expr})
            return Match()
        else:
            if not isinstance(expr, OperandsTuple):
                expr = OperandsTuple((expr,))
            if pattern.head and not all(isinstance(e, pattern.head) for e in expr):
                return False
            if pattern.condition and not all(pattern.condition(e) for e in expr):
                return False
            if pattern.name:
                return Match({pattern.name: expr})
            return Match()
    elif isinstance(pattern, NamedPattern):
        name, p =  pattern.operands
        m = match(p, expr)
        if m:
            return m + Match({name: expr })
        return False
    elif isinstance(pattern, Operation):
        if isinstance(expr,Operation) and type(pattern) is type(expr):
            return match(PatternTuple(pattern.operands), OperandsTuple(expr.operands))
    else:
        return Match() if pattern == expr else False










class Wildcard(Expression):
    single = 1
    one_or_more= 2
    zero_or_more = 3

    name = ""
    mode = single
    head = None
    condition = None

    _hash = None

    def __init__(self, name = "", mode = single, head = None, condition = None):
        self.name = name
        self.mode = mode
        self.head = head
        self.condition = condition

    def __str__(self):
        if isinstance(self.head, tuple):
            head_string = "({})".format("|".join(h.__name__ for h in self.head))
        elif self.head is not None:
            head_string = self.head.__name__
        else:
            head_string = ""
        return "{}{}{}{}".format(self.name,
                                 "_" * self.mode,
                                 head_string,
                                 self.condition.__name__ if self.condition else "")

    def __repr__(self):
        if isinstance(self.head, tuple):
            head_string = "({})".format(", ".join(h.__name__ for h in self.head))
        elif self.head is not None:
            head_string = self.head.__name__
        else:
            head_string = "None"

        return "Wildcard({}, {}, {}, {})".format(repr(self.name),
                                                 "_" * self.mode,
                                                 head_string,
                                                 self.condition.__name__ if self.condition else "None")

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.name == other.name
                and self.head == other.head
                and self.mode == other.mode
                and self.condition == other.condition)

    def __hash__(self):
        if not self._hash:
            self._hash = hash((self.name, self.mode, self.head, self.condition))
        return self._hash


class Match(dict):
    """
    Subclass of dict that overloads the + operator
    to create a new dictionary combining the entries.
    It fails when there are duplicate keys.
    """

    def __add__(self, other):
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
            # make sure the sets of keys are disjoint
        overlap = set(self.keys()) & set(other.keys())
        if not len(overlap) == 0:
            raise ValueError()

        ret = Match(self)
        ret.update(other)
        return ret

#    def __repr__(self):
#        return "Match({})".format(dict.__repr__(self))
#
#    def __str__(self):
#        return "Match({})".format(dict.__str__(self))

    def __bool__(self):
        return True

    __nonzero__ = __bool__



import re
name_mode_pattern = re.compile(r"^([A-Za-z]?[A-Za-z0-9]*)(_{0,3})$")

def wc(name_mode = "_", head = None, condition = None):
    m = name_mode_pattern.match(name_mode)
    if not m:
        raise ValueError()
    name, modelength = m.groups()
    mode = len(modelength) or Wildcard.single
    if not 1 <= mode <= 3:
        raise ValueError()
    return Wildcard(name, mode=mode , head=head, condition=condition)


from itertools import izip
from types import FunctionType, MethodType

def make_classmethod(method, cls):
    return MethodType(method, cls, type(cls))

def decorate_classmethod(method, clsmethodname = "create", setup = None):
    def decorator(dcls):
        clsmtd = getattr(dcls, clsmethodname).im_func
        def dclsmtd(cls, *args):
            return method(dcls, clsmtd, cls, *args)

        dclsmtd.method = method
        dclsmtd.decorated = clsmtd
        dclsmtd.dcls = dcls
        dclsmtd.decorators = (method,) + getattr(clsmtd, "decorators", ())
        dclsmtd.__name__ = clsmethodname

        nmtd = make_classmethod(dclsmtd, dcls)
#        assert nmtd.im_func is dclsmtd
        setattr(dcls, clsmethodname, nmtd)
#        if method.__name__.endswith("_mtd"):
#            cls_flag = "_" + method.__name__[:-4]
#        if cls_flag:
#            setattr(dcls, cls_flag, True)
        return dcls
    return decorator



def flat_mtd(dcls, clsmtd, cls, *ops):
    nops = sum(((o,) if not isinstance(o,cls) else o.operands for o in ops),())
    return clsmtd(cls, *nops)

flat = decorate_classmethod(flat_mtd)


def idem_mtd(dcls, clsmtd, cls, *ops):
    return clsmtd(cls, *sorted(set(ops), key = cls.order_key))

idem = decorate_classmethod(idem_mtd)


def orderless_mtd(dcls, clsmtd, cls, *ops):
    return clsmtd(cls, *sorted(ops, key = cls.order_key))
orderby = decorate_classmethod(orderless_mtd)


unequals = lambda x: (lambda y: x != y)

def filter_neutral_mtd(dcls, clsmtd, cls, *ops):
    c_n = cls.neutral_element
    if len(ops) == 0:
        return c_n
    fops = tuple(filter(unequals(c_n), ops))
    if len(fops) > 1:
        return clsmtd(cls, *fops)
    elif len(fops) == 1:
        # the remaining operand is the single non-trivial one
        return fops[0]
    else:
        # the original list of operands consists only of neutral elements
        return ops[0]

filter_neutral = decorate_classmethod(filter_neutral_mtd)


CLS = object()
DCLS = object()

def extended_isinstance(obj, class_info, dcls, cls):
    if isinstance(class_info, tuple):
        return any(extended_isinstance(obj, cli, dcls, cls) for cli in class_info)
    if class_info is CLS:
        class_info = cls
    elif class_info is DCLS:
        class_info = dcls
    return isinstance(obj, class_info)

def check_signature_mtd(dcls, clsmtd, cls, *ops):
    sgn = cls.signature
    if not len(ops) == len(sgn):
        raise WrongSignature()
    if not all(extended_isinstance(o, s, dcls, cls) for o, s in izip(ops, sgn)):
        raise WrongSignature()
    return clsmtd(cls, *ops)
check_signature = decorate_classmethod(check_signature_mtd)


def check_signature_flat_mtd(dcls, clsmtd, cls, *ops):
    sgn = cls.signature[0]
    if not all(extended_isinstance(o, sgn, dcls, cls) for o in ops):
        raise WrongSignature()
    return clsmtd(cls, *ops)
check_signature_flat = decorate_classmethod(check_signature_flat_mtd)


def match_replace_binary_mtd(dcls, clsmtd, cls, *ops):

    rules = cls.binary_rules
    j = 1
    while j < len(ops):
        first, second = ops[j-1], ops[j]
        m = False
        r = False
        for patterns, replacement in rules:
            m = match(PatternTuple(patterns), OperandsTuple((first, second)))
            if m:
                try:
                    r = replacement(**m)
                    break
                except CannotSimplify:
                    continue

        if r is not False:
            # if Operation is also "flat", then expand out the operands of a binary-simplified result
            if flat_mtd in getattr(cls.create.im_func, "decorators",()) and isinstance(r, cls):
                ops = ops[:j-1] + r.operands + ops[j+1:]
            else:
                ops = ops[:j-1] + (r,) + ops[j+1:]
            if j > 1:
                j -= 1
        else:
            j += 1

    return clsmtd(cls, *ops)

match_replace_binary = decorate_classmethod(match_replace_binary_mtd)


def match_replace_mtd(dcls, clsmtd, cls, *ops):
    rules = cls.rules
    ops = OperandsTuple(ops)
    r = False
    for patterns, replacement in rules:
        m = match(PatternTuple(patterns), ops)

        if m:
            try:
                return replacement(**m)
            except CannotSimplify:
                continue
    return clsmtd(cls, *ops)

match_replace = decorate_classmethod(match_replace_mtd)
