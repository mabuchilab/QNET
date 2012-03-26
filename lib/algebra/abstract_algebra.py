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



def n(obj):
    try:
        # print obj, obj.n()
        return obj.n()
    except AttributeError:
        return obj

def is_number(obj):
    return isinstance(obj, (int, long, float, complex))

def is_integral(obj):
    return isinstance(obj, (int, long))
    

def set_union(*sets):
    return reduce(lambda set1, set2: set1 | set2, sets, set(()))

def set_intersection(*sets):
    return reduce(lambda set1, set2: set1 & set2, sets, set(()))
    
def simplify(obj, **rules):
    try:
        return obj.simplify(**rules)
    except AttributeError:
        return obj


def NotImplementedFn(*args):
    """Always return NotImplemented"""
    return NotImplemented

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

class OperandsError(AlgebraError):
    """
    Raised when the operands of an operation violate a specific rule.
    """
    def __init__(self, msg, *operands):
        AlgebraError.__init__(self, msg + ": " + str(map(repr, operands)))

class CannotCompare(AlgebraError):
    """
    Raised when comparison between two objects is not implemented by a certain function.
    """
    pass

class SymbolIdentifierTaken(AlgebraError):
    """
    Raised on an attempt to create a new symbol with an existing identifier.
    """
    def __init__(self, existing_symbol, existing_args, new_args):
        AlgebraError.__init__(self, "A symbol with identifier %s was already defined (%s).\n Newly requested args: %r" 
                                        % (existing_symbol.identifier, existing_symbol, new_args))



class Expression(object):
    """Abstract base class defining the basic methods any Expression object should implement."""
    
    # def expand(self):
    #     """Expand out sums distributively."""
    #     return self
    
    def substitute(self, var_map):
        """
        Substitute symbols for other expressions via
        var_map = {Symbol_1_object:SubstituteExpression_1,... }
        and return self with all substitutions carried out.
        """
        return self
        
    subs = substitute
        
    def __call__(self, **kwargs):
        """
        Substitute some values for symbols into the expression and subsequently call evalf(.) for the result.
        """
        return evalf(self.substitute(kwargs))
    
    def evalf(self):
        """
        Force evaluation of all expressions where this is possible.
        """
        return self
    
    def tex(self):
        """
        Return a string containing a TeX-representation of self.
        """
        return str(self)
        
    def _latex_(self):
        # print "called _latex_ on %s" % self
        return self.tex()
        
        
    # def diff(self, wrt):
    #     """
    #     Differentiate expression with respect to 'wrt'.
    #     """
    #     # as long as there exist no symbolic derivatives, this is fine
    #     if self.free_of(wrt):
    #         return 0
    #         
    #     raise NotImplementedError(self.__class__.__name__)
    
    def free_of(self, *symbols):
        """
        Check if expression is independent of all passed symbols
        """
        raise NotImplementedError(self.__class__.__name__)
        
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
    
    def __cmp__(self, other):
        """
        Return
            -1 if self < other,
            0 if self == other,
            +1 if other > self.
            
        A strict and well-defined (i.e. unique) order relation on all expressions is important
        for any kind of implementation of a 'canonical form' of an expression.
        """
        if self is other:
            return 0
        
        s_alg = algebra(self)
        o_alg = algebra(other)
        
        # if they have the same algebra
        if s_alg is o_alg:
            # compare within the algebra
            return s_alg.cmp_within_algebra(self, other)
            
        # otherwise compare their algebras
        return cmp(s_alg, o_alg)
    
    def simplify(self):
        raise NotImplementedError(self.__class__.__name__)


# def expand(expr):
#     """
#     (Safe) expand: Distributively expand out all products between sums.
#     """
#     try:
#         return expr.expand()
#     except AttributeError:
#         return expr


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

def evalf(expr):
    """
    (Safe) evalf: Evaluate all (sub-) expressions that can be converted to pure numerical form.
    """
    try:
        return expr.evalf()
    except AttributeError:
        return expr


def tex(obj):
    """
    Return a LaTeX string-representation of obj.
    """
    if isinstance(obj, str):
        return identifier_to_tex(obj)
    if is_number(obj):
        return format_number_for_tex(obj)
    if isinstance(obj, SympyBasic):
        # TODO fix bracketing
        return sympy_latex(obj)[1:-1] #trim '$' at beginning and end of returned string
    try:
        return obj.tex()
    except AttributeError:
        try:
            return obj._latex_()
        except AttributeError:
            return str(obj)
            
    
# def diff(expr, wrt, n = 1):
#     """
#     Safe differentiation.
#     """
#     if n < 0:
#         raise AlgebraError()
#     if n == 0:
#         return expr
#     try:
#         return diff(expr.diff(wrt), wrt, n - 1)
#     except AttributeError:
#         return 0
        
def free_of(expr, *symbols):
    """
    Safe free_of
    """
    try:
        return expr.free_of(*symbols)
    except AttributeError:
        return True
        
def symbols(expr):
    """
    Safe symbols
    """
    try:
        return expr.symbols()
    except AttributeError:
        try:
            return expr.atoms()
        except AttributeError:
            return set(())
    
    
def format_number_for_tex(num):
    if num == 0: #also True for 0., 0j
        return "0"
    if isinstance(num, complex):
        if num.real == 0:
            if num.imag == 1:
                return "i"
            if num.imag == -1:
                return "(-i)"
            if num.imag < 0:
                return "(-%si)" % format_number_for_tex(-num.imag)
            return "%si" % format_number_for_tex(num.imag)
        if num.imag == 0:
            return format_number_for_tex(num.real)
        return "(%s + %si)" % (format_number_for_tex(num.real), format_number_for_tex(num.imag))
    if num < 0:
        return "(%g)" % num
    return "%g" % num

greek_letter_strings = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", \
                        "zeta", "eta", "theta", "vartheta", "iota", "kappa", \
                        "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", \
                        "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", \
                        "varphi", "chi", "psi", "omega", \
                        "Gamma", "Delta", "Theta", "Lambda", "Xi", \
                        "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]

import re
def identifier_to_tex(identifier):
    """
    If an identifier contains a greek symbol name as a separate word,
    (e.g. 'my_alpha_1' contains 'alpha' as a separate word, but 'alphaman' doesn't)
    add a backslash in front.
    """
    identifier = reduce(lambda a,b: "{%s_%s}" % (b, a), ["{%s}" % part for part in reversed(identifier.split("__"))])
    p = re.compile(r'([^\\A-Za-z]?)(%s)\b' % "|".join(greek_letter_strings))
    return p.sub(r'\1{\\\2}', identifier)




class Symbol(Expression):
    """
    Abstract base class for Symbolic objects.
    All Symbols (across all algebras) are uniquely identified by an identifier string.
    
    Once a symbol of a given identifier has been created, an additional attempt
    to create a symbol of the same identifier will either
        - yield the same object, only if it is being instantiated with exactly the same additional arguments
        - lead to a 'SymbolIdentifierTaken' exception.
    """
    
    symbol_cache = {}
        
    def __new__(cls, identifier, *args):
        # perform cache lookup
        # if not hasattr(cls, 'symbol_cache'):
        #     cls.symbol_cache = {}
        instance, i_args = cls.symbol_cache.get(identifier, (False, None))
        if instance:
            if args != i_args:
                raise SymbolIdentifierTaken(instance, i_args, args)
                
            return instance
        instance = Expression.__new__(cls)
        cls.symbol_cache[identifier] = instance, args
        return instance
    
    def __init__(self, identifier, *args):
        self._identifier = identifier
        self._hash = hash((self.__class__, identifier, args))
        self._args = args
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        """
        For any identifier (and other parametric args)
        there exists only a single instance,
        equality implies actual instance equality
        """
        return self is other
    
    # def expand(self):
    #     return self
    
    def substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        if self.identifier in var_map:
            return var_map[self.identifier]
        return self
    
    def __repr__(self):
        return "%s%r" % (self.__class__.__name__, (self._identifier,) + self._args)
    
    def evalf(self):
        return self
    
    def simplify(self, **rules):
        return self
    
    def tex(self):
        return identifier_to_tex(self._identifier)
        
    def free_of(self, *symbols):
        return not self in symbols and not self.identifier in symbols
    
    def symbols(self):
        return set([self])
    
    @property
    def identifier(self):
        return self._identifier
    

CHECK_OPERANDS = False


class insert_default_rules(object):
    
    def __init__(self, default_rules):
        self.default_rules = default_rules
    
    def __call__(self, function_to_decorate):
        # print function_to_decorate
        def function_with_default_rules(*args, **additional_rules):
            # print args, additional_rules
            rules = self.default_rules.copy()
            rules.update(additional_rules)
            return function_to_decorate(*args, **rules)

        return function_with_default_rules
    
    

class Operation(Expression):
    """
    Abstract base class for all operations,
    where the operands themselves are also expressions.
    """
    @classmethod
    def check_operands(cls, *operands, **rules):
        raise NotImplementedError(cls.__name__)

    ##### instance methods/properties
    def __init__(self, *operands, **rules):
        """
        Create a symbolic operation with the operands. No filtering/simplification
        of operands is performed at this stage, but if the global var CHECK_OPERANDS
        is True, the operands are checked according to the rules.
        """
        if CHECK_OPERANDS:
            self.__class__.check_operands(*operands, **rules)
            
        self._operands = operands
        
        if len(operands) == 0:
            raise Exception('Need at least one operand')
            
        self._hash = None   #lazy hash creation, i.e. when it is actually needed
    
    @property
    def operands(self):
        return self._operands
        
    
    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.__class__, self.operands))
        return self._hash
    
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(map(repr, self.operands)))
    
    # def expand(self):
    #     return self.__class__.apply_with_rules(*map(expand, self.operands))
    
    def substitute(self, var_map):
        return self.__class__.apply_with_rules(*map(lambda o: substitute(o, var_map), self.operands))
    
    def evalf(self):
        return self.__class__.apply_with_rules(*map(evalf, self.operands))
    
    def __eq__(self, other):
        return type(self) == type(other) and self.operands == other.operands
        
    def symbols(self):
        return set_union(*[symbols(op) for op in self.operands])
        
    def free_of(self, *symbols):
        return all((free_of(op, *symbols) for op in self.operands))
    
    def simplify(self, **rules):
        """
        Simplify the operation according to some rules by reapplying it 
        (with these rules) to the simplified operands.
        """
        s_ops = []

        for o in self.operands:
            try:
                s_ops.append(simplify(o, **rules))

            except CannotSimplify:
                s_ops.append(o)
        
        return self.apply_with_rules(*s_ops, **rules)
        
    
    
    @classmethod
    def apply_with_rules(cls, *operands, **rules):
        """
        Instead of directly instantiating an operation object, 
        it should always be instantiated by applying the operations with
        all (default) rules. This ensures that no invalid expressions are created.
        """
        raise NotImplementedError(cls.__name__)

        
             

PRINT_PRETTY = False

class MultiaryOperation(Operation):
    """Base class for all operations that require two or more operands."""

    ## TODO clean up operation/filter_operands call interface
    
    

        
    operation_symbol = ' # ' #overwrite this!!!
    
    tex_symbol = ' \diamond '
    
    default_rules = {}
    
    
    def __str__(self):
        return "(%s)" % self.operation_symbol.join(map(str, self.operands))
            
    def tex(self):
        return "\left( %s \\right)" % self.tex_symbol.join(map(tex, self.operands))
    
    
    @classmethod
    @insert_default_rules(default_rules)
    def check_operands(cls, *operands, **rules):
        if len(operands) < 2:
            raise OperandsError('Need at least two operands', *operands)
    
    @classmethod
    def filter_operands(cls, operands):
        return operands

    @classmethod
    def handle_zero_operands(cls):
        raise NotImplementedError("Not implemented yet for class %s" % cls.__name__)
    
    @classmethod
    def handle_single_operand(cls, operand):
        return operand
    

    @classmethod
    @insert_default_rules(default_rules)    
    def apply_with_rules(cls, *operands, **rules):
        
        operands = cls.filter_operands(operands)
        
        if len(operands) == 0:
            return cls.handle_zero_operands()
            
        elif len(operands) == 1:
            return cls.handle_single_operand(operands[0])
        
        return cls(*operands, **rules)


class AssociativeOperation(MultiaryOperation):
    
    #RULES inherited from MultiaryOperation
    default_rules = MultiaryOperation.default_rules.copy()
    
    #plus new rule
    default_rules.update(expand_associatively = True)

    
    def __str__(self):
        if PRINT_PRETTY:
            return "\n(%s\n)" % ("\n    " + self.operation_symbol).join([str(op).replace("\n", "\n    ") for op in self.operands])
        return "(%s)" % (self.operation_symbol).join([str(op)for op in self.operands])    
    
    def __repr__(self):
        if PRINT_PRETTY:
            return "%s(\n    %s\n)" % (self.__class__.__name__, ",\n    ".join([repr(op).replace("\n", "\n    ") for op in self.operands]))
        return "%s(%s)" % (self.__class__.__name__, ",".join([repr(op) for op in self.operands]))
    
    @classmethod
    def check_associatively_expanded(cls, *operands):
        if any((isinstance(op, cls) for op in operands)):
            raise OperandsError("Not expanded", *operands)
    
    @classmethod
    @insert_default_rules(default_rules)
    def check_operands(cls, *operands, **rules):
        MultiaryOperation.check_operands.im_func(cls, *operands, **rules)
        
        if rules.get('expand_associatively', cls.default_rules.get('expand_associatively')):
            cls.check_associatively_expanded(*operands)
        
    @classmethod
    @insert_default_rules(default_rules)
    def apply_with_rules(cls, *operands, **rules):
        
        expand_associatively = rules['expand_associatively']
        
        if expand_associatively:
            operands = cls.expand_associatively(operands)
            
        operands = cls.filter_operands(operands)
        operands = cls.combine_binary(*operands, **rules)
        
        # if expand_associatively:
        #     operands = cls.expand_associatively(operands)
        

        if len(operands) == 0:
            return cls.handle_zero_operands()
        elif len(operands) == 1:
            return cls.handle_single_operand(operands[0])

        return cls(*operands, **rules)
    
    
    @classmethod
    @insert_default_rules(default_rules)
    def combine_binary(cls, *operands, **rules):
        for i, (lhs, rhs) in enumerate(izip(operands[:-1], operands[1:])):
            try:
                combined = cls.simplify_binary(lhs, rhs, **rules)
                
                # allow for the possibility that the simplified expression 
                # is still a composite object of the type cls, be careful to raise the CannotSimplify exception,
                # i.e., within simplify_binary NEVER return cls(lhs,rhs) as this would lead to an infinite loop!!!
                
                combined_operands = (combined,) if not isinstance(combined, cls) else combined.operands
                
                if i > 0: #maybe the combined object can now be combined with the previous operand
                    return operands[ : i-1] +  cls.combine_binary(operands[i-1], *(combined_operands + operands[i + 2 : ]), **rules)
                    
                return operands[ : i] +  cls.combine_binary(*(combined_operands + operands[i + 2 : ]), **rules)
            except CannotSimplify:
                pass
        return operands
    
    @classmethod
    def simplify_binary(cls, lhs, rhs, **rules):
        if not hasattr(cls, 'has_been_warned'):
            print "WARNING: please implement this correctly for class %s" % cls.__name__
            cls.has_been_warned = True
        raise CannotSimplify
        
    @classmethod
    def expand_associatively(cls, operands):
        operands = operands
        for i, o in enumerate(operands):
            if isinstance(o, cls):
                return operands[ : i] + o.operands + cls.expand_associatively(operands[i + 1 : ])
        return operands
    

def expand(obj):
    try:
        return obj.expand()
    except AttributeError:
        return obj


                
        

class Addition(AssociativeOperation):
    
    operation_symbol = ' + '
    tex_symbol = operation_symbol
        
    def evalf(self):
        return reduce(lambda a, b: a + b, map(evalf, self.operands))
    
    @classmethod
    def filter_operands(cls, operands):
        operands = filter(lambda o: o != 0, operands)
        return sorted(operands)
    
    @classmethod
    def handle_zero_operands(cls):
        return 0
        
    # def diff(self, wrt):
    #     return sum([diff(op, wrt) for op in self.operands])
    
    def expand(self):
        return self.apply_with_rules(*map(expand, self.operands))

class Multiplication(AssociativeOperation):
    operation_symbol = ' * '
    tex_symbol = ' '
    
    @classmethod
    def filter_operands(cls, operands):
        return filter(lambda op: op != 1, operands)
        
    @classmethod
    def handle_zero_operands(cls):
        return 1
    
    
    def expand(self):
        operands = map(expand, self._operands)
        for i, o in enumerate(operands):
            if isinstance(o, Addition):
                pre_factor = operands[:i]
                post_factor = operands[i+1:]
                new_summands = map(lambda f: self.__class__.apply_with_rules(*(pre_factor + [f] + post_factor)), o._operands)
                return o.__class__.apply_with_rules(*new_summands).expand()
        return self.__class__.apply_with_rules(*operands)
    
    # def diff(self, wrt):
    #     result = 0
    #     for i, op in self.operands:
    #         result += (self.__class__.apply_with_rules(*(self.operands[i:] + (diff(op, wrt),) + operands[i+1:])))
    #     return result
    
    def evalf(self):
        return reduce(lambda a, b: a * b, map(evalf, self.operands))



class BinaryOperation(MultiaryOperation):
    
    @property
    def lhs(self):
        return self._operands[0]
    
    @property
    def rhs(self):
        return self._operands[1]
    
    def __init__(self, lhs, rhs, **rules):
        Operation.__init__(self, lhs, rhs, **rules)
        
    @classmethod
    def apply_with_rules(cls, lhs, rhs, **rules):
        return MultiaryOperation.apply_with_rules.im_func(cls, lhs, rhs, **rules)


class Power(BinaryOperation):
    default_rules = {'combine_power_powers': True,         # (a**b)**c = a**(b*c)
                    'expand_fraction_powers': True,        # (a/b)**c = a**c/b**c
                    }
    
    operation_symbol = '**'
    
    
    @classmethod
    @insert_default_rules(default_rules)
    def check_operands(cls, base, exponent, **rules):
        if base == 1 or base == 0 or exponent == 1 or exponent == 1:
            raise OperandsError('', base, exponent)
    
    @classmethod
    @insert_default_rules(default_rules)
    def apply_with_rules(cls, base, exponent, **rules):
        
        if is_number(base) and is_number(exponent):
            return base ** exponent
            
        if base == 1:
            return 1
            
        if exponent == 0:
            return 1
            
        if exponent == 1:
            return base
        
            
        if isinstance(base, cls):
            return cls(base.base, base.exponent * exponent)
            
        if isinstance(base, CoefficientTermProduct):
            return cls.apply_with_rules(base.coeff, exponent) * cls.apply_with_rules(base.term, exponent)
        
            
        return cls(base, exponent, **rules)
    
    def tex(self):
        return "{%s}^{%s}" % (tex(self.base), tex(self.exponent))
    
    def __str__(self):
        return "%s**%s" % (self.base, self.exponent)
    
    @property
    def base(self):
        return self._operands[0]
    
    @property
    def exponent(self):
        return self._operands[1]

    
    # def diff(self, wrt):
    #     """The exponent is always assumed to be a constant integer, so it is not differentiated."""
    #     base, exponent = self.operands
    # 
    #     d_b = diff(base, wrt)
    #     
    #     return exponent * base**(exponent - 1) * d_b
    
        
    def evalf(self):        
        return evalf(self.base)**evalf(self.exponent)


class CoefficientTermProduct(BinaryOperation):
    """
    Abstract class to implement a cross-algebra product: coeff * term
    """
    
    operation_symbol = ' * '
    tex_symbol = " "
    
    
    @property
    def coeff(self):
        return self._operands[0]
    
    @property
    def term(self):
        return self._operands[1]
    
    def expand(self):
        # this only expands the term, not the coefficient
        coeff, term = expand(self.coeff), expand(self.term)
        
        if isinstance(term, Addition): #only expand for term
            return term.__class__.apply_with_rules(*(map( lambda t: self.__class__.apply_with_rules(coeff, t), term.operands)))
        return self.__class__.apply_with_rules(coeff, term)
    
    # def diff(self, wrt):
    #     return diff(self.coeff, wrt) * self.term + self.coeff * diff(self.term, wrt) 




# class Fraction(BinaryOperation):
#     """
#     Abstract superclass for fractions:
#     numerator/denominator == Fraction(numerator, denominator).
#     """
#     
#     operation_symbol = " / "
#     
#     @property
#     def numerator(self): #zaehler
#         return self._operands[0]
#     
#     @property
#     def denominator(self):#nenner
#         return self._operands[1]
#     
#     def tex(self):
#         return "\\frac{%s}{%s}" % (tex(self.numerator), tex(self.denominator))
#         
#     
#     def diff(self, wrt):
#         return self.__class__.apply_with_rules(diff(self.numerator, wrt) * self.denominator - self.numerator * diff(self.denominator), self.denominator**2)
#     
#     def expand(self):
#         return self.__class__.apply_with_rules(expand(self.numerator), self.denominator)

class UnaryOperation(Operation):
    """
    Abstract superclass for all unary operations,
    i.e. operations with a single operand.
    """
    
    @classmethod
    def check_operands(cls, *operands, **rules):
        if len(operands) != 1:
            raise OperandsError('Wrong number of operands', *operands)
    
    def __init__(self, operand, **rules):
        Operation.__init__(self, operand, **rules)
    
    @property
    def operand(self):
        return self.operands[0]


class Number:
    """Dummy class to assign an algebra to Python built-in numbers."""
    pass



    

def algebra(obj):
    """
    Return the associated algebra of obj.
    
    obj may be:
        1) a python number:
            algebra(5) == algebra(1.) == algebra(1+0j) == Number
        2) an instance of a subclass of Algebra:
            algebra(ScalarSymbol) == Scalar
        3) a subclass of Algebra:
            algebra(ScalarSymbol) == algebra(Scalar) == Scalar
    """
    if isinstance(obj, Algebra) or isinstance(obj, AlgebraType):
        return obj.algebra
    if isinstance(obj, SympyBasic):
        # print "sympyBasic:", obj
        return SympyBasic
    if is_number(obj):
        return Number
    if isinstance(obj, ndarray):
        return None
    try:
        sympify(obj)
        # print 'sympifiable:', obj
        return Number #enforce sympification
    except SympifyError:
        pass
    
    return None


class AlgebraType(type):
    """
    This serves as a Metaclass for Algebra classes.
    See http://en.wikibooks.org/wiki/Python_Programming/MetaClasses
    for some information what metaclasses actually do.
    In our case, we would like to use it only to achieve that any direct Algebra
    subclass (e.g. Scalar) gets a class property called 'algebra' referencing itself.
    Subclasses of the Algebra subclasses (e.g. ScalarAddition) then simply inherit this
    class property.    
    Example
    
        class MyAlgebra(Algebra):
            pass
    
        class MyAlgebraSubclass(MyAlgebra):
            pass
    
        MyAlgebra.algebra == MyAlgebra            # True, because MyAlgebra is direct subclass of Algebra
        MyAlgebraSubclass.algebra == MyAlgebra    # True, because this class attribute is inherited from MyAlgebra
    
    """
    def __new__(clsmeta, clsname, clssuper, clsdict):
        """
        Called when a new class is defined with the name clsname, 
        the superclasses clssuper, and the properties and methods defined in clsdict.
        clsmeta points to the metaclass, i.e. to AlgebraType (or a submetaclass of this).
        """
        
        #create the actual class
        cls = type.__new__(clsmeta, clsname, clssuper, clsdict)
        
        # If we're creating the Algebra class itself, then skip ahead.
        if clsname != 'Algebra': 
            
            # if the new class is a direct subclass of Algebra
            if Algebra in clssuper:
                
                #store a self reference of the class in the 'private' property _algebra
                cls._algebra = cls
                
            else:
                #otherwise make sure that it is derived off some other Algebra
                if not any((issubclass(sup, Algebra) for sup in clssuper)):
                    print clsname, clssuper
                    raise Exception()
                
        return cls
    
    
    # make a read-only public property for the specific algebra class itself
    @property
    def algebra(cls):
        return cls._algebra


def make_binary_operation_method(operation_map_name, operation_string = ''):
    """
    Return a method for a binary operation between to algebraic expressions.
    """
    
    def operation(a, b):
        operation_map = getattr(algebra(a), operation_map_name, {})
        # print operation_map_name[:-4], a,b
        return operation_map.get(algebra(b), NotImplementedFn)(a, b)
        
    if operation_string != '':
        operation.__doc__ = """
        To implement the operation '%s' 
        for the specific algebras of 'self' and of 'other',
        implement the operation in a function
            >> def operation(self, other):
            >>     # ...
        and add an entry to the appropriate operation_map:
            >> self.algebra.%s[algebra(other)] = operation
        """ % (operation_string, operation_map_name)
    return operation


##### TODO: QUICK FIX!
old_cmp = cmp
def cmp(a, b):
    try:
        return old_cmp(a,b)
    except TypeError:
        if isinstance(a, complex) or isinstance(b, complex):
            return old_cmp(a.real, b.real) or old_cmp(a.imag, b.imag)
        raise TypeError('Woops')


            
class Algebra(object):
    """
    Basis class from which all Algebras should directly be derived.
    Every direct subclass of Algebra has an attribute 'algebra' that points to itself.
    For more information on this see the documentation for AlgebraType.
    
    To mathematical operations for all MyAlgebra objects can be changed at runtime
    by changing the according map property.
    To implement the Addition of two MyAlgebra objects, this code would work:
    
        def add_my_algebra_objects(a,b):
            # do and return something ...
        
        MyAlgebra.add_map[MyAlgebra] = add_my_algebra_objects
    
    and if we now also wanted to implement addition with another algebra such 
    as python numbers, we need to specify both addition from left and from right:
        
        def add_my_algebra_object_to_number(obj, number):
            # do and return something
        
        # notice the order of the arguments!
        # this is consistent with how one would implement
        # the __radd__ method.
        def add_number_to_my_algebra_object(obj, number):
            # do and return something (else?)
        
        MyAlgebra.add_map[Number] = add_my_algebra_object_to_number
        MyAlgebra.radd_map[Number] = add_number_to_my_algebra_object
        
        
    This kind of mechanism is implemented 
    for all of the binary operators listed in the first column.
        +   add_map     radd_map
        -   sub_map     rsub_map
        /   div_map     rdiv_map
        *   mul_map     rmul_map
        **  pow_map     rpow_map
        <<  lshift_map  rlshift_map
        >>  rshift_map  rrshift_map
    
    The second and third column specify the names which the operation dictionary 
    should have as algebra class properties for operating from left and from the right.
    Once you have specified the mathematical operation between MyAlgebra via 
    the operation maps of MyAlgebra, you need not re-implement this behaviour 
    within the other Algebra class.
    This way, only one Algebra has to 'know about' the other.
    E.g. Operators should know Scalars, but not vice versa.
    """
    
    
    # specify that the type of the Algebra class is actually the AlgebraType
    __metaclass__ = AlgebraType
    
    @property
    def algebra(self):
        """
        In contrast to the property of the class itself, 
        this is a property of each instance of an algebra object.
        """
        return self.__class__._algebra
        
    
    
    __mul__ = make_binary_operation_method('mul_map', 'self * other')
    __rmul__ = make_binary_operation_method('rmul_map', 'other * self')
    
    __add__ = make_binary_operation_method('add_map', 'self + other')
    __radd__ = make_binary_operation_method('radd_map', 'other + self')
    
    __sub__ = make_binary_operation_method('sub_map', 'self - other')
    __rsub__ = make_binary_operation_method('rsub_map', 'other - self')
    
    
    
    __div__ = make_binary_operation_method('div_map', 'self / other')
    __rdiv__ = make_binary_operation_method('rdiv_map', 'other / self')
    
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    
    __pow__ = make_binary_operation_method('pow_map', 'self**other')
    __rpow__ = make_binary_operation_method('rpow_map', 'other**self')
    
    __lshift__ = make_binary_operation_method('lshift_map', 'self << other')
    __rlshift__ = make_binary_operation_method('rlshift_map', 'other << self')
    
    __rshift__ = make_binary_operation_method('rshift_map', 'self >> other')
    __rrshift__ = make_binary_operation_method('rrshift_map', 'other >> self')
    
    def cmp_within_algebra(self, other):
        """
        General comparison function between objects of the same algebra.
        See also the documentation of Expression.__cmp__
        
        
        Algebra Subclasses that feature expression that aren't subclasses of
        Symbol, CoefficientTermProduct or Operation, need to overload this method.
        If the overloaded implementation calls this method, it should make sure,
        that any additional types of expressions must come AFTER all of the 
        above-mentioned ones.
        """
        
        """
        for any type of coefficientTermProduct:
        compare terms first and only if they are equal, compare coefficients
        """
        if isinstance(self, CoefficientTermProduct):
            if isinstance(other, CoefficientTermProduct):
                return cmp(self.algebra, other.algebra) \
                        or cmp(self.term, other.term) \
                        or cmp(self.coeff, other.coeff)
            return cmp(self.term, other) or cmp(self.coeff, 1)
        if isinstance(other, CoefficientTermProduct):
            return cmp(self, other.term) or cmp(1, other.coeff)
            
        """
        For any type of power, compare bases first and then exponents
        """
        if isinstance(self, Power):
            if isinstance(other, Power):
                return cmp(self.algebra, other.algebra) \
                    or cmp(self.base, other.base) \
                    or cmp(self.exponent, other.exponent)
            return cmp(self.base, other)
        if isinstance(other, Power):
            return cmp(self, other.base)
        
        
        
        """
        Among symbols, compare the class and then the identifier.
        Individual symbols should come before anything else.
        """
        if isinstance(self, Symbol):
            if isinstance(other, Symbol):
                return  cmp(self.__class__.__name__, other.__class__.__name__) \
                        or cmp(self.identifier, other.identifier)
            return -1
        if isinstance(other, Symbol):
            return +1
        
        """
        Among operations, compare the class and then the operands.
        """
        if isinstance(self, Operation):
            if isinstance(other, Operation):
                return cmp(self.__class__.__name__, other.__class__.__name__) \
                        or cmp(self.operands, other.operands)
            return -1
        
        if isinstance(other, Operation):
            return +1
        
        
        raise CannotCompare(repr((self, other)))
    
