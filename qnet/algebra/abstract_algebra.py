#!/usr/bin/env python
# encoding: utf-8
"""
abstract_algebra.py

Created by Nikolas Tezak on 2011-02-17.
Copyright (c) 2011 . All rights reserved.
"""

from __future__ import division
from itertools import izip



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


class Expression(object):
    """
    Basic class defining the basic methods any Expression object should implement.
    """

    def substitute(self, var_map):
        """
        Substitute symbols for other expressions via
        var_map = {Symbol_1_object:SubstituteExpression_1,... }
        and return self with all substitutions carried out.
        """
        return self._substitute(var_map)

    def _substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        return self

    def tex(self):
        """
        Return a string containing a TeX-representation of self.
        """
        return self._tex()

    def _tex(self):
        return str(self)

    def _repr_latex_(self):
        """
        For compatibility with the IPython notebook, generate TeX expression and surround it with $'s.
        """
        return "$%s$" % self.tex()

#    def free_of(self, *symbols):
#        """
#        Check if expression is independent of all passed symbols
#        """
#        return len(set(symbols) & self.symbols()) == 0


    def symbols(self):
        """
        Return set of symbols contained within the expression.
        """
        return self._symbols()

    def _symbols(self):
        raise NotImplementedError(self.__class__.__name__)

#    def mathematica(self):
#        return self._mathematica()
##        raise NotImplementedError(self.__class__.__name__)


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


def substitute(expr, var_map):
    """
    (Safe) substitute: Substitute objects for symbols
    """
    try:
        return expr.substitute(var_map)
    except AttributeError:
        return expr

#        try:
#            # try substituting sympy_objects
#            var_map = dict((k,v) for (k,v) in var_map.items() if not isinstance(k, Expression) and not isinstance(v, Expression))
#            return expr.subs(var_map)
#        except AttributeError:
#            return expr
#        except Exception, e:
#            print "Warning while trying to substitute in %s an error occured: %s" % (expr, e)
#            return expr

def tex(obj):
    """
    Return a LaTeX string-representation of obj.
    """
    if isinstance(obj, str):
        return identifier_to_tex(obj)
    if is_number(obj):
        return format_number_for_tex(obj)
#    if isinstance(obj, SympyBasic):
#        return sympy_latex(obj)#[1:-1] #trim '$' at beginning and end of returned string
    try:
        return obj.tex()
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


def set_union(*sets):
    """
    Similar to sum(), but for sets. Generate the union of an arbitrary number of set arguments.
    :param sets: Sets to for the union of.
    :type sets: set
    :return: Union set.
    :rtype: set
    """
    return reduce(lambda a, b: a.union(b), sets, set(()))


class Operation(Expression):
    """
    Abstract base class for all operations,
    where the operands themselves are also expressions.
    """
    # hash str, is generated on demand (lazily)
    _hash = None

    def __init__(self, *operands):
        """
        Create a symbolic operation with the operands. No filtering/simplification
        of operands is performed at this stage.
        """
        self._operands = operands

    @property
    def operands(self):
        return self._operands

    def _symbols(self):
        return set_union(*[symbols(op) for op in self.operands])

    def _substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        return self.__class__.create(*map(lambda o: substitute(o, var_map), self.operands))

    def _tex(self):
        return r"{{\rm {}}}\left({{}}\right)".format(self.__class__.__name__, ", ".join(o.tex() for o in self.operands))


    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(map(repr, self.operands)))
    
    # def mathematica(self):
    #     return "%s[%s]" % (self.__class__.__name__, ", ".join(map(mathematica, self.operands)))

    def __eq__(self, other):
        return type(self) == type(other) and self.operands == other.operands

    def __hash__(self):
        if not self._hash:
            self._hash = hash((self.__class__, self.operands))
        return self._hash

    @classmethod
    def create(cls, *operands):
        """
        Instead of directly instantiating an Operation object,
        it should always be instantiated by applying the operations with
        all (default) rules. This ensures that no invalid expressions are created.
        """
        return cls(*operands)

    @classmethod
    def order_key(cls, a):
        """
        Provide a default ordering mechanism for achieving canonical ordering of expressions sequences.
        """
        if hasattr(a, "_key_"):
            return a._key_()
        return a


    def _key_(self):
        return (self.__class__.__name__,) + self.operands

        
########################################################################################################################
########################### WILDCARDS AND PATTERN MATCHING FUNCTIONS ###################################################
########################################################################################################################



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
    """
    Specialized tuple to store expression operands for the purpose of matching them to patterns.
    """
    def __getitem__(self, item):
        if isinstance(item, slice):
            return OperandsTuple(super(OperandsTuple, self).__getitem__(item))
        return super(OperandsTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return OperandsTuple(super(OperandsTuple, self).__getslice__(i,j))

class PatternTuple(tuple):
    """
    Specialized tuple to store expression pattern operands.
    """
    def __getitem__(self, item):
        if isinstance(item, slice):
            return OperandsTuple(super(PatternTuple, self).__getitem__(item))
        return super(PatternTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return PatternTuple(super(PatternTuple, self).__getslice__(i,j))


class NamedPattern(Operation):
    """
    Create a named (sub-)pattern for later use in processing elements of a matched expression.
        NamedPattern(name, pattern)

    :param name: Pattern identifier
    :type name: str
    :param pattern: Pattern expression
    :type pattern:  Expression
    """

    def __init__(self, name, pattern):
        super(NamedPattern, self).__init__(name, pattern)

def trace(fn):
    """
    Function decorator to receive debugging information about function calls and return values.
    :param fn: Function whose calls to trace
    :type fn: FunctionType
    :return: Decorated function
    :rtype: FunctionType
    """

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
    """
    Helper method to flatten out PatternTuple and OperandTuple elements within a sequence.
    :param seq: Sequence of objects, some of which may be PatternTuple or OperandTuple objects.
    :type seq: sequence
    :return: A flattened list.
    :rtype: list
    """
    sres = []
    for s in seq:
        if isinstance(s, (PatternTuple, OperandsTuple)):
            sres += list(s)
        else:
            sres.append(s)
    return sres

#@trace
def update_pattern(expr, match_obj):
    """
    Replace all wildcards in the pattern expression with their matched values as specified in a Match object.
    :param expr: Pattern expression
    :type expr: Expression
    :param match_obj: Match object
    :type match_obj: Match
    :return: Expression with replaced wildcards
    :rtype: Expression
    """
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

    """
    Match a pattern against an expression and return a Match object if successful or False, if not.
    Works recursively.
    :param pattern: Pattern expression
    :type pattern: (Expression, PatternTuple)
    :param expr: Expression to match against the pattern.
    :type expr: (Expression, OperandsTuple)
    :return: Match object or False
    :rtype: Match or False
    """
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
    """
    Basic wildcard expression that can match a single expression or in the context of
    matching the operands of an Operation object one may match one_or_more or zero_or_more operands
    with the same wildcards. If the wildcard has a name, a successful match leads to a Match object in which the
    object that matched the wildcard is stored under that name. One can also restrict the type of the matched Expression
    by providing a head argument and the condition argument allows for passing a function that performs additional tests
    on a potential match.
    """
    single = 1
    one_or_more= 2
    zero_or_more = 3

    name = ""
    mode = single
    head = None
    condition = None

    _hash = None

    def __init__(self, name = "", mode = single, head = None, condition = None):
        """
        :param name: Wildcard name, default ""
        :type name: str
        :param mode: The matching mode, i.e. how many objects/operands can the wildcard match.
        :type mode: One of Wildcard.single, Wildcard.one_or_more, Wildcard.zero_or_more
        :param head: Restriction of the type of the matched expression
        :type head: tuple of class objects or None
        :param condition: An additional function that returns True if match should be accepted.
        :type condition: FunctionType or None
        """
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
    """
    Helper function to create a Wildcard object.
    :param name_mode: Combined name and mode (see Wildcard doc) argument.
        * "A" -> name="A", mode = Wildcard.single
        * "A_" -> name="A", mode = Wildcard.single
        * "B__" -> name="B", mode = Wildcard.one_or_more
        * "B___" -> name="C", mode = Wildcard.zero_or_more
    :type name_mode: str
    :param head: See Wildcard doc
    :type head: tuple of class objects or None
    :param condition:  See Wildcard doc
    :type condition: FunctionType or None
    :return: A Wildcard object
    :rtype: Wildcard
    """
    m = name_mode_pattern.match(name_mode)
    if not m:
        raise ValueError()
    name, modelength = m.groups()
    mode = len(modelength) or Wildcard.single
    if not 1 <= mode <= 3:
        raise ValueError()
    return Wildcard(name, mode=mode , head=head, condition=condition)


########################################################################################################################
########################### CLASS DECORATORS TO ACHIEVE OPERAND PREPROCESSING ##########################################
########################################################################################################################

from itertools import izip
from types import FunctionType, MethodType

def make_classmethod(method, cls):
    return MethodType(method, cls, type(cls))

def preprocess_create_with(method):

    def decorator(dcls):
        clsmtd = getattr(dcls, "create").im_func
        def dclsmtd(cls, *args):
            return method(dcls, clsmtd, cls, *args)

        dclsmtd.method = method
        dclsmtd.decorated = clsmtd
        dclsmtd.dcls = dcls
        dclsmtd.__doc__ = clsmtd.__doc__
        dclsmtd.decorators = (method,) + getattr(clsmtd, "decorators", ())
        dclsmtd.__name__ = "create"

        nmtd = make_classmethod(dclsmtd, dcls)
        setattr(dcls, "create", nmtd)
        return dcls

    return decorator



def flat_mtd(dcls, clsmtd, cls, *ops):
    nops = sum(((o,) if not isinstance(o,cls) else o.operands for o in ops),())
    return clsmtd(cls, *nops)

flat = preprocess_create_with(flat_mtd)


def idem_mtd(dcls, clsmtd, cls, *ops):
    return clsmtd(cls, *sorted(set(ops), key = cls.order_key))

idem = preprocess_create_with(idem_mtd)


def orderless_mtd(dcls, clsmtd, cls, *ops):
    return clsmtd(cls, *sorted(ops, key = cls.order_key))
orderby = preprocess_create_with(orderless_mtd)


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

filter_neutral = preprocess_create_with(filter_neutral_mtd)


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
check_signature = preprocess_create_with(check_signature_mtd)


def check_signature_flat_mtd(dcls, clsmtd, cls, *ops):
    sgn = cls.signature[0]
    if not all(extended_isinstance(o, sgn, dcls, cls) for o in ops):
        raise WrongSignature()
    return clsmtd(cls, *ops)
check_signature_flat = preprocess_create_with(check_signature_flat_mtd)


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

match_replace_binary = preprocess_create_with(match_replace_binary_mtd)


def match_replace_mtd(dcls, clsmtd, cls, *ops):
    for patterns, replacement in cls.rules:
        m = match(PatternTuple(patterns), OperandsTuple(ops))
        if m:
            try:
                return replacement(**m)
            except CannotSimplify:
                continue
    return clsmtd(cls, *ops)

match_replace = preprocess_create_with(match_replace_mtd)

