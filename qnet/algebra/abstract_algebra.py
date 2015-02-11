# coding=utf-8
# This file is part of QNET.
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

r"""
Abstract Algebra
================

The abstract algebra package provides a basic interface
    for defining custom Algebras.

See :ref:`abstract_algebra` for more details.

"""
from __future__ import division
from functools import reduce
from abc import ABCMeta, abstractmethod
from types import MethodType
import six

import sys

if six.PY3:
    basestring = str
    long = int


def _trace(fn):
    """
    Function decorator to receive debugging information about function calls and return values.

    :param fn: Function whose calls to _trace
    :type fn: FunctionType
    :return: Decorated function
    :rtype: FunctionType
    """

    ### uncomment for debugging
    # def _tfn(*args, **kwargs):
    #     print("[", "-" * 40)
    #     ret = fn(*args, **kwargs)
    #     print("{}({},{}) called".format(fn.__name__, ", ".join(repr(a) for a in args),
    #                                     ", ".join(str(k) + "=" + repr(v) for k, v in kwargs.items())))
    #     print("-->", repr(ret))
    #     print("-" * 40, "]")
    #     return ret
    # return _tfn

    return fn




# define our own exceptions/errors
class AlgebraException(Exception):
    """
    Base class for all errors concerning the mathematical
        definitions and rules of an algebra.
    """
    pass


class AlgebraError(AlgebraException):
    """
    Base class for all errors concerning the mathematical
        definitions and rules of an algebra.
    """
    pass


class CannotSimplify(AlgebraException):
    """
    Raised when an expression cannot be further simplified
    """
    pass


class WrongSignatureError(AlgebraError):
    """
    Raise when an operation is instantiated
        with operands of the wrong signature.
    """
    pass

@six.add_metaclass(ABCMeta)
class Expression(object):
    """
    Basic class defining the basic methods any Expression object should implement.
    """

    def substitute(self, var_map):
        """
        Substitute all_symbols for other expressions.

        :param var_map: Dictionary with entries of the form ``{symbol: substitution}``
        :type var_map: dict
        """
        return self._substitute(var_map)

    def _substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        return self

    def tex(self):
        """
        Return a string containing a TeX-representation of self.
        Note that this needs to be wrapped by '$' characters for 'inline' LaTeX use.
        """
        return self._tex()

    @abstractmethod
    def _tex(self):
        return str(self)

    def _repr_latex_(self):
        """
        For compatibility with the IPython notebook, generate TeX expression and surround it with $'s.
        """
        return "${}$".format(self.tex())

    def all_symbols(self):
        """
        :return: The set of all_symbols contained within the expression.
        :rtype: set
        """
        return self._all_symbols()

    @abstractmethod
    def _all_symbols(self):
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def __hash__(self):
        """
        Provide a hashing mechanism for self.
        """
        raise NotImplementedError(self.__class__.__name__)

    def __eq__(self, other):
        """
        Implements a very strict definition of ``self == other``.
        This should be overloaded where appropriate.
        """
        if self is other:
            return True
        return False

    def __ne__(self, other):
        """
        If it is well-defined (i.e. boolean), simply return
        the negation of ``self.__eq__(other)``
        Otherwise return NotImplemented.
        """
        eq = self.__eq__(other)
        if type(eq) is bool:
            return not eq
        return NotImplemented


def substitute(expr, var_map):
    """
    (Safe) substitute, substitute objects for all symbols.

    :param expr: The expression in which to perform the substitution
    :param var_map: The substitution dictionary. See :py:meth:`qnet.algebra.abstract_algebra.substitute` documentation
    :type var_map: dict

    """
    try:
        return expr.substitute(var_map)
    except AttributeError:
        if expr in var_map:
            return var_map[expr]
        return expr


def tex(obj):
    """
    :param obj: Object to represent in LaTeX.
    :return: Return a LaTeX string-representation of obj.
    :rtype: str
    """
    try:
        return obj.tex()
    except AttributeError:
        return r"{{\rm {!s}}}".format(obj)


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
#    return creduce(lambda a, b: a.replace(b, b[0].upper() + b[1:]), words, string)
#
#

#def free_of(expr, *all_symbols):
#    """
#    Safe free_of
#    """
#    try:
#        return expr.free_of(*all_symbols)
#    except AttributeError:
#        return True
#
#def all_symbols(expr):
#    """
#    Safe all_symbols
#    """
#    try:
#        return expr.all_symbols()
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
#    identifier = creduce(lambda a,b: "{%s_%s}" % (b, a), ["{%s}" % part for part in reversed(identifier.split("__"))])
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
#    identifier = creduce(lambda a,b: "Subscript[%s,%s]" % (b, a), reversed(identifier.split("__")))
#    p = re.compile(r'\b(%s)\b' % "|".join(greek_letter_strings))
#    repl = lambda m:  r"\[" + greekToLatex[m.group(1)] + "]"
#    return p.sub(repl, identifier)


class KeyTuple(tuple):
    def __lt__(self, other):
        # print("<", self, other)
        if isinstance(other, (long, basestring)):
            return False
        if isinstance(other, KeyTuple):
            return super(KeyTuple, self).__lt__(other)
        raise AlgebraException("Cannot compare: {}".format(other))

    def __gt__(self, other):
        # print(">", self, other)
        if isinstance(other, (long, basestring)):
            return True
        if isinstance(other, KeyTuple):
            return super(KeyTuple, self).__gt__(other)
        raise AlgebraException("Cannot compare: {}".format(other))




def set_union(*sets):
    """
    Similar to ``sum()``, but for sets. Generate the union of an arbitrary number of set arguments.

    :param sets: Sets to for the union of.
    :type sets: set
    :return: Union set.
    :rtype: set
    """
    return reduce(lambda a, b: a.union(b), sets, set(()))


def all_symbols(expr):
    """
    Return all all_symbols featured within an expression.

    :param expr: The expression to find all_symbols in.
    :return: A set of all_symbols within expr.
    :rtype: set
    """
    try:
        return expr.all_symbols()
    except AttributeError:
        return set(())


class Operation(Expression):
    """
    Abstract base class for all operations,
    where the operands themselves are also expressions.
    """

    # hash str, is generated on demand (lazily)
    __hash = None

    def __init__(self, *operands):
        """
        Create a symbolic operation with the given operands.::

            Operation(*operands)

        :param operands: The operands of the expression.
        :type operands: object or as defined in the Operation's signature
        """
        self._operands = operands

    @property
    def operands(self):
        """
        :return: The operands of the operation.
        :rtype: tuple
        """
        return self._operands

    def _all_symbols(self):
        return set_union(*[all_symbols(op) for op in self.operands])


    def _substitute(self, var_map):
        if self in var_map:
            return var_map[self]
        return self.__class__.create(*map(lambda o: substitute(o, var_map), self.operands))

    def _tex(self):
        return r"{{\rm {}}}\left({{}}\right)".format(self.__class__.__name__, ", ".join(tex(o) for o in self.operands))


    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join(map(repr, self.operands)))

    # def mathematica(self):
    #     return "%s[%s]" % (self.__class__.__name__, ", ".join(map(mathematica, self.operands)))

    def __eq__(self, other):
        # print(type(self), type(other), type(self) == type(other))
        return type(self) == type(other) and self.operands == other.operands

    def __hash__(self):
        if not self.__hash:
            self.__hash = hash((self.__class__, self.operands))
        return self.__hash

    @classmethod
    def create(cls, *operands):
        """
        Instead of directly instantiating an instance of any subclass of Operation,
        it is advised to call the ``create()`` classmethod instead.
        This method takes the same arguments as the constructor, but can preprocess them and even return an object
        of a different type based on the operands.

        :param operands: The operands for the operation.
        """
        return cls(*operands)

    @classmethod
    def order_key(cls, obj):
        """
        Provide a default ordering mechanism for achieving canonical ordering of expressions sequences.

        :param obj: The object to create a key for.
        """
        try:
            return obj._order_key()
        except AttributeError:
            return str(obj)


    def _order_key(self):
        return KeyTuple((self.__class__.__name__,) + tuple(map(Operation.order_key, self.operands)))


mathematica = lambda s: s



########################################################################################################################
########################### WILDCARDS AND PATTERN MATCHING FUNCTIONS ###################################################
########################################################################################################################



inf = float('inf')

@_trace
def match_range(pattern):
    """
    Compute how many objects/operands a given pattern can minimally and maximally match.

    :param pattern: The pattern object
    :return: min_number, max_number
    :rtype: tuple
    :raise: ValueError, if unknown pattern mode for Wildcard object
    """
    if isinstance(pattern, Wildcard):
        if pattern.mode == Wildcard.single:
            return 1, 1
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
            #noinspection PyTypeChecker
            return OperandsTuple(super(OperandsTuple, self).__getitem__(item))
        return super(OperandsTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return OperandsTuple(super(OperandsTuple, self).__getslice__(i, j))


class PatternTuple(tuple):
    """
    Specialized tuple to store expression pattern operands.
    """

    def __getitem__(self, item):
        if isinstance(item, slice):
            #noinspection PyTypeChecker
            return PatternTuple(super(PatternTuple, self).__getitem__(item))
        return super(PatternTuple, self).__getitem__(item)

    def __getslice__(self, i, j):
        return PatternTuple(super(PatternTuple, self).__getslice__(i, j))


class NamedPattern(Operation):
    """
    Create a named (sub-)pattern for later use in processing elements of a matched expression.::

        NamedPattern(name, pattern)

    :param name: Pattern identifier
    :type name: str
    :param pattern: Pattern expression
    :type pattern:  Expression, PatternTuple
    """

    def __init__(self, name, pattern):
        super(NamedPattern, self).__init__(name, pattern)





def _flatten(seq):
    """
    Helper method to _flatten out PatternTuple and OperandTuple elements within a sequence.

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


@_trace
def update_pattern(expr, match_obj):
    """
    Replace all wildcards in the pattern expression with their matched values as specified in a Match object.

    :param expr: Pattern expression
    :type expr: Expression or PatternTuple
    :param match_obj: Match object
    :type match_obj: Match
    :return: Expression with replaced wildcards
    :rtype: Expression or PatternTuple
    """
    if isinstance(expr, Wildcard):
        if expr.name in match_obj:
            return match_obj[expr.name]
    elif isinstance(expr, (PatternTuple, OperandsTuple)):
        return expr.__class__(_flatten([update_pattern(o, match_obj) for o in expr]))
    elif isinstance(expr, Operation):
        return expr.__class__(*_flatten([update_pattern(o, match_obj) for o in expr.operands]))

    return expr


@_trace
def match(pattern, expr):
    """
    Match a pattern against an expression and return a Match object if successful or False, if not.
    Works recursively.

    :param pattern: Pattern expression
    :type pattern: Expression or PatternTuple
    :param expr: Expression to match against the pattern.
    :type expr: Expression or OperandsTuple
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
            if isinstance(p0, Wildcard):
                if p0.mode != Wildcard.single:
                    a0, b0 = match_range(p0)
                    for k in range(a0, min(l, b0) + 1):
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
                            #                    print(m0, update_pattern(prest, m0), mrest)
                        if mrest:
                            return m0 + mrest
                    return False
            else:
                # noinspection PyTypeChecker
                m0 = match(p0, expr[0])
                if m0:
                    orest = expr[1:]
                    if len(m0):
                        mrest = match(update_pattern(prest, m0), orest)
                    else:
                        mrest = match(prest, orest)
                    #                    print(m0, update_pattern(prest, m0), mrest)
                    if mrest:
                        return m0 + mrest
                return False

    elif isinstance(pattern, Wildcard):
        if pattern.mode == Wildcard.single:
            if isinstance(expr, OperandsTuple):
                assert len(expr) == 1
                #noinspection PyRedeclaration
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
        name, p = pattern.operands
        m = match(p, expr)
        if m:
            return m + Match({name: expr})
        return False
    elif isinstance(pattern, Operation):
        if isinstance(expr, Operation) and type(pattern) is type(expr):
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
    """Value of :py:attr:`Wildcard.mode` for matching single operands/objects"""

    one_or_more = 2
    """Value of :py:attr:`Wildcard.mode` for matching one or more operands/objects"""

    zero_or_more = 3
    """Value of :py:attr:`Wildcard.mode` for matching zero or more operands/objects"""

    name = ""
    """name/identifier of the wildcard (default = "")."""

    mode = single
    """mode of the wildcard, i.e. how many operands it can match (default = ``single``)."""

    head = None
    """head/type of the matched object (default = ``None``, corresponding to no restriction)."""

    condition = None
    """extra condition for a successful match (default = ``None``, corresponding to no restriction)."""

    _hash = None

    #noinspection PyRedeclaration
    def __init__(self, name="", mode=single, head=None, condition=None):
        """
        :param name: Wildcard name, (default = "")
        :type name: str
        :param mode: The matching mode, i.e. how many objects/operands can the wildcard match.
        :type mode: One of :py:attr:`Wildcard.single`, :py:attr:`Wildcard.one_or_more`, :py:attr:`Wildcard.zero_or_more`
        :param head: Restriction of the type of the matched expression
        :type head: tuple or type or None
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

    def _tex(self):
        return r"{\rm " + self.name + ("\_" * self.mode) + self.head.__name__ + (
            "?{}".format(self.condition.__name__) if hasattr(self.condition, "__name__") else "") + "}"

    def _all_symbols(self):
        return set(())


class Match(dict):
    """
    Subclass of dict that overloads the + operator
    to create a new dictionary combining the entries.
    It fails when there are duplicate keys.
    """

    def __add__(self, other):
        if not len(self):
            return other
        if not len(other):
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

    # noinspection PyMethodMayBeStatic
    def __bool__(self):
        return True

    __nonzero__ = __bool__


import re

name_mode_pattern = re.compile(r"^([A-Za-z]?[A-Za-z0-9]*)(_{0,3})$")


def wc(name_mode="_", head=None, condition=None):
    """
    Helper function to create a Wildcard object.

    :param name_mode: Combined name and mode (cf :py:class:`Wildcard`) argument.

        * ``"A"`` -> ``name="A", mode = Wildcard.single``
        * ``"A_"`` -> ``name="A", mode = Wildcard.single``
        * ``"B__"`` -> ``name="B", mode = Wildcard.one_or_more``
        * ``"B___"`` -> ``name="C", mode = Wildcard.zero_or_more``

    :type name_mode: str
    :param head: See Wildcard doc
    :type head: tuple or ClassType or None
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
    return Wildcard(name, mode=mode, head=head, condition=condition)


########################################################################################################################
########################### CLASS DECORATORS TO ACHIEVE OPERAND PREPROCESSING ##########################################
########################################################################################################################



def make_classmethod(method, cls):
    """
    Make a bound classmethod from an unbound method taking an additional first argument ``cls``

    :param method: The unbound method
    :type method: FunctionType
    :param cls: The class to bind it to
    :type cls: type
    :return: Bound class method
    :rtype: MethodType
    """
    return MethodType(method, cls, type(cls))


def preprocess_create_with(method):
    """
    This factory method allows for adding argument pre-processing decorators to a class's ``create`` classmethod.

    :param method: A decorating create classmethod ``f()`` with signature:
            ``f(decorated_class, decorated_method, cls, *args)``
    :type method: FunctionType
    :return: A class decorator function that decorates the 'create' classmethod of the decorated class.
    :rtype: FunctionType
    """

    # noinspection PyDocstring
    def decorator(dcls):
        if six.PY2:
            clsmtd = getattr(dcls, "create").im_func
        else:
            clsmtd = getattr(dcls, "create").__func__

        # noinspection PyDocstring
        def dclsmtd(cls, *args):
            return method(dcls, clsmtd, cls, *args)

        dclsmtd.method = method
        dclsmtd.decorated = clsmtd
        dclsmtd.dcls = dcls
        dclsmtd.__doc__ = (str(clsmtd.__doc__)
                           + "\n-- {}.create() preprocessed by {} --\n".format(dcls.__name__, method.__name__)
                           + str(method.__doc__))

        # store a list of all applied decorators as an attribute of the new create method's im_func.
        dclsmtd.decorators = (method,) + getattr(clsmtd, "decorators", ())
        dclsmtd.__name__ = "create"

        # noinspection PyTypeChecker
        # nmtd = make_classmethod(dclsmtd, dcls)
        nmtd = classmethod(dclsmtd)
        setattr(dcls, "create", nmtd)
        return dcls

    # Copy docstring from method to class decorator
    decorator.__doc__ = """
    {}

    Automatically generated class decorator based on the method ``qnet.algebra.abstract_algebra.{}()`` using
    :py:func:`preprocess_create_with`.
""".format(method.__doc__, method.__name__)

    return decorator


#noinspection PyUnusedLocal,PyDocstring
def _assoc(dcls, clsmtd, cls, *ops):
    """
    Associatively expand out nested arguments of the flat class.

        >>> @assoc
        >>> class Plus(Operation):
        >>>     pass
        >>> Plus.create(1,Plus(2,3))
            Plus(1,2,3)
    """
    nops = sum(((o,) if not isinstance(o, cls) else o.operands for o in ops), ())
    return clsmtd(cls, *nops)


# noinspection PyTypeChecker
assoc = preprocess_create_with(_assoc)


#noinspection PyUnusedLocal,PyDocstring
def _idem(dcls, clsmtd, cls, *ops):
    """
    Remove duplicate arguments and order them via the cls's order_key key object/function.
    E.g.

        >>> @idem
        >>> class Set(Operation):
        >>>     pass
        >>> Set.create(1,2,3,1,3)
            Set(1,2,3)
    """
    return clsmtd(cls, *sorted(set(ops), key=cls.order_key))


# noinspection PyTypeChecker
idem = preprocess_create_with(_idem)


#noinspection PyUnusedLocal,PyDocstring
def _orderby(dcls, clsmtd, cls, *ops):
    """
    Re-order arguments via the class's ``order_key`` key object/function.
    Use this for commutative operations:
    E.g.

        >>> @orderby
        >>> class Times(Operation):
        >>>     pass
        >>> Times.create(2,1)
            Times(1,2)
    """
    try:
        return clsmtd(cls, *sorted(ops, key=cls.order_key))
    except TypeError as te:
        print(list(map(cls.order_key,ops)))
        raise te


# noinspection PyTypeChecker
orderby = preprocess_create_with(_orderby)

unequals = lambda x: (lambda y: x != y)


#noinspection PyUnusedLocal,PyDocstring
def _filter_neutral(dcls, clsmtd, cls, *ops):
    """
    Remove occurrences of a neutral element from the argument/operand list, if that list has at least two elements.
    To use this, one must also specify a neutral element,
    which can be anything that allows for an equality check with each argument.
    E.g.

        >>> @filter_neutral
        >>> class X(Operation):
        >>>     neutral_element = 1
        >>> X.create(2,1,3,1)
            X(2,3)

    """
    c_n = cls.neutral_element
    if not len(ops):
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


# noinspection PyTypeChecker
filter_neutral = preprocess_create_with(_filter_neutral)

CLS = object()
DCLS = object()


def extended_isinstance(obj, class_info, dcls, cls):
    """
    Like isinstance but with two extra arguments to allow for placeholder objects ``(DCLS, CLS)``
    to stand for the class objects passed as extra arguments ``dcls``, ``cls``.
    This allows one to specify a self-referential
    `signature` class attribute to allow for recursive Operation signatures.
    E.g.

        >>> @check_signature
        >>> class X(Operation):
        >>>     signature = str, X

    will yield an exception, because X within the class body refers to a class object that has not been defined yet.
    Instead, one can do

        >>> @check_signature
        >>> class X(Operation):
        >>>     signature = str, CLS

    to refer to the class of the object being instantiated (could be a subclass of ``X``), or

        >>> @check_signature
        >>> class X(Operation):
        >>>     signature = str, DCLS

    to always refer to ``X`` itself and not a subclass.

    :param obj: The instance
    :type obj: object
    :param class_info: A type, ``DCLS``, ``CLS``, or a tuple of these
    :type class_info: type or tuple of type-objects
    :param dcls: The (super-)class that the signature is defined for.
    :type dcls: type
    :param cls: The concrete (sub-)class whose instance is being initialized.
    :type cls: type
    """
    if isinstance(class_info, tuple):
        return any(extended_isinstance(obj, cli, dcls, cls) for cli in class_info)
    if class_info is CLS:
        #noinspection PyRedeclaration
        class_info = cls
    elif class_info is DCLS:
        #noinspection PyUnusedLocal
        class_info = dcls
    return isinstance(obj, class_info)


#noinspection PyDocstring
def _check_signature(dcls, clsmtd, cls, *ops):
    """
    Check that the operands passed to the create classmethod of an Operation type conform to certain types.
    For each allowed argument/operand, provide a tuple of types (or one of ``CLS, DCLS``, see extended_isinstance docs).
    E.g.

        >>> @check_signature
        >>> class X(Operation):
        >>>     signature = str, (int, str)
        >>>
        >>> X.create("1", 2)
            X("1", 2)
        >>> X.create("1", "2")
            X("1", "2")

    The following all raise :py:class:`WrongSignatureError` exception.

        >>> X.create("1")
        >>> X.create(1, "1")
        >>> X.create("1", 2, 3)

    """
    sgn = cls.signature
    if not len(ops) == len(sgn):
        raise WrongSignatureError()
    if not all(extended_isinstance(o, s, dcls, cls) for o, s in zip(ops, sgn)):
        raise WrongSignatureError("class: {}, operands: {}".format(str(cls), str(ops)))
    return clsmtd(cls, *ops)


check_signature = preprocess_create_with(_check_signature)


#noinspection PyDocstring
def _check_signature_assoc(dcls, clsmtd, cls, *ops):
    """
    Like :py:func:`check_signature` but for :py:func:`assoc`-decorated Operations. In this case the signature need only contain a single entry.

        >>> @assoc
        >>> @check_signature
        >>> class X(Operation):
        >>>     signature = str
        >>> X.create("hello", "you")
            X("hello", "you")

    The following then raises a :class:`WrongSignatureError` because the third argument is no string

        >>> X.create("hello", "you", 2)

    """
    sgn = cls.signature[0]
    if not all(extended_isinstance(o, sgn, dcls, cls) for o in ops):
        print(sgn, dcls, cls, ops)
        raise WrongSignatureError()
    return clsmtd(cls, *ops)


# noinspection PyTypeChecker
check_signature_assoc = preprocess_create_with(_check_signature_assoc)


# noinspection PyDocstring
# noinspection PyUnusedLocal
def _match_replace(dcls, clsmtd, cls, *ops):
    """
    Match and replace a full operand specification to a function that provides a replacement for the whole expression
    or raises a :py:class:`CannotSimplify` exception.
    E.g.

    First define wildcards:

        >>> A = wc("A")
        >>> A_float = wc("A", head = float)


    Then an operation:

        >>> @match_replace
        >>> class Invert(Operation):
        >>>     _rules = []

    Then some _rules:

        >>> Invert._rules += [
        >>>     ((Invert(A),), lambda A: A),
        >>>     ((A_float,), lambda A: 1./A),
        >>> ]

    Check rule application:

        >>> Invert.create("hallo")              # matches no rule
            Invert("hallo")
        >>> Invert.create(Invert("hallo"))      # matches first rule
            "hallo"
        >>> Invert.create(.2)                   # matches second rule
            5.

    A pattern can also have the same wildcard appear twice:

        >>> @match_replace
        >>> class X(Operation):
        >>>     _rules = [
        >>>         ((A, A), lambda A: A),
        >>>     ]

        >>> X.create(1,2)
            X(1,2)

        >>> X.create(1,1)
            1

    """

    for patterns, replacement in cls._rules:
        m = match(PatternTuple(patterns), OperandsTuple(ops))
        if m:
            try:
                return replacement(**m)
            except CannotSimplify:
                continue
    return clsmtd(cls, *ops)


# noinspection PyTypeChecker
match_replace = preprocess_create_with(_match_replace)


#noinspection PyDocstring
# noinspection PyUnusedLocal
def _match_replace_binary(dcls, clsmtd, cls, *ops):
    """
    Similar to :py:func:`match_replace`, but for arbitrary length operations, such that each two pairs of subsequent operands are matched pairwise.

        >>> A = wc("A")

        >>> @match_replace_binary
        >>> class FilterDupes(Operation):
        >>>     _rules = [
        >>>         ((A,A), lambda A: A),
        >>>     ]

        >>> FilterDupes.create(1,2,3,4)         # No subsequent duplicates present
            FilterDupes(1,2,3,4)

        >>> FilterDupes.create(1,2,2,3,4)       # Some duplicates
            FilterDupes(1,2,3,4)

    Note that this only works for *subsequent* duplicate entries:

        >>> FilterDupes.create(1,2,3,2,4)       # Some duplicates, but not subsequent
            FilterDupes(1,2,3,2,4)
    """
    _rules = cls._binary_rules
    j = 1
    while j < len(ops):
        first, second = ops[j - 1], ops[j]
        m = False
        r = False
        for patterns, replacement in _rules:
            m = match(PatternTuple(patterns), OperandsTuple((first, second)))
            if m:
                try:
                    r = replacement(**m)
                    break
                except CannotSimplify:
                    continue

        if not(r is False):
            # if Operation is also "assoc", then expand out the operands of a binary-simplified result
            if _assoc in getattr(cls.create.im_func
                                 if six.PY2
                                 else cls.create.__func__,
                                 "decorators", ()) and isinstance(r, cls):
                ops = ops[:j - 1] + r.operands + ops[j + 1:]
            else:
                ops = ops[:j - 1] + (r,) + ops[j + 1:]
            if j > 1:
                j -= 1
        else:
            j += 1

    return clsmtd(cls, *ops)


match_replace_binary = preprocess_create_with(_match_replace_binary)


def singleton(cls):
    """
    Singleton class decorator. Turns a class object into a unique instance.

    :param cls: Class to decorate
    :type cls: type
    :return: The singleton instance of that class
    :rtype: cls
    """

    # noinspection PyDocstring
    class S(cls):
        __instance = None

        def __hash__(self):
            return hash(cls)

        # noinspection PyMethodMayBeStatic
        def _symbols(self):
            return set(())

        def __repr__(self):
            return cls.__name__

        def __call__(self):
            return self.__instance

    S.__name__ = cls.__name__
    S.__instance = s = S()

    return s


def prod(sequence, neutral=1):
    """
    Analog of the builtin `sum()` method.
    :param sequence: Sequence of objects that support being multiplied to each other.
    :type sequence: Any object that implements __mul__()
    :param neutral: The initial return value, which is also returned for zero-length sequence arguments.
    :type neutral: Any object that implements __mul__()
    :return: The product of the elements of `sequence`
    """
    return reduce(lambda a, b: a * b, sequence, neutral)
