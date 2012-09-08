## coding=utf-8
#"""
#superoperator_algebra.py
#
#The specification of a quantum mechanical symbolic operator algebra.
#Basic elements from which expressions can be built are operator all_symbols and locally acting operators.
#Each operator has an associated `space` property which gives the Hilbert space on which it acts non-trivially.
#In order to not have to specify all degrees of freedom in advance, an operator is assumed to act as the identity on
#all degrees of freedom that are independent of its space, as is customary in the physics literature.
#
#    >>> x = OperatorSymbol("x", "1")
#    >>> x.space
#    LocalSpace("1",...)
#
#The available local operator types are
#
#    Create(localspace)
#    Destroy(localspace)
#    LocalSigma(localspace, j, k)
#
#There exist some useful constants to specify neutral elements of Operator addition and multiplication:
#    ZeroOperator
#    OperatorIdentity
#
#Quantum Operator objects can be added together in code via the infix '+' operator and multiplied with the infix '*' operator.
#They can also be added to or multiplied by scalar objects.
#In the first case, the scalar object is multiplied by the IdentityOperator constant.
#
#Operations involving at least one quantum Operator argument are
#
#    OperatorPlus(A, B, C, ...)
#    OperatorTimes(A, B, C, ...)
#    ScalarTimesOperator(coefficient, term)
#    Adjoint(op)
#    PseudoInverse(op)
#
#"""
##TODO UPDATE DOCSTRING
#from operator_algebra import *
#
#class SuperOperator(object):
#    """
#    The basic operator class, which fixes the abstract interface of operator objects
#    and where possible also defines the default behavior under operations.
#    Any operator contains an associated HilbertSpace object,
#    on which it is taken to act non-trivially.
#    """
#
#    __metaclass__ = ABCMeta
#
#    # which data types may serve as scalar coefficients
#    scalar_types = Operator.scalar_types
#
#
#    @property
#    def space(self):
#        """
#        The Hilbert space associated with the operator on which it acts non-trivially
#        """
#        return self._space
#
#    @abstractproperty
#    def _space(self):
#        raise NotImplementedError(self.__class__.__name__)
#
#    def superadjoint(self):
#        """
#        :return: The Hermitian adjoint of the operator.
#        :rtype: Operator
#        """
#        return self._superadjoint()
#
#    def _superadjoint(self):
#        return SuperAdjoint.create(self)
#
#
##    def pseudo_inverse(self):
##        """
##        :return: The pseudo-Inverse of the Operator, i.e., it inverts the operator on the orthogonal complement of its nullspace
##        :rtype: Operator
##        """
##        return self._pseudo_inverse()
##
##    @abstractmethod
##    def _pseudo_inverse(self):
##        return PseudoInverse.create(self)
#
#    def to_qutip(self, full_space = None):
#        """
#        Create a numerical representation of the superoperator as a QuTiP object.
#        Note that all symbolic scalar parameters need to be replaced by numerical values before calling this method.
#        :param full_space: The full Hilbert space in which to represent the superoperator.
#        :type full_space: HilbertSpace
#        :return: The matrix representation of the superoperator.
#        :rtype: qutip.Qobj
#        """
#        if full_space is None:
#            full_space = self.space
#        return self._to_qutip(full_space)
#
#    @abstractmethod
#    def _to_qutip(self, full_space):
#        raise NotImplementedError(str(self.__class__))
#
#    def expand(self):
#        """
#        Expand out distributively all products of sums. Note that this does not expand out sums of scalar coefficients.
#        :return: A fully expanded sum of superoperators.
#        :rtype: SuperOperator
#        """
#        return self._expand()
#
#    @abstractmethod
#    def _expand(self):
#        raise NotImplementedError(self.__class__.__name__)
#
#    def series_expand(self, param, about, order):
#        """
#        Expand the superoperator expression as a truncated power series in a scalar parameter.
#        :param param: Expansion parameter.
#        :type param: sympy.core.symbol.Symbol
#        :param about: Point about which to expand.
#        :type about:  Any one of Operator.scalar_types
#        :param order: Maximum order of expansion.
#        :type order: int >= 0
#        :return: tuple of length (order + 1) featuring the contributions at different orders in the parameters
#        :rtype: tuple of SuperOperator
#        """
#        return self._series_expand(param, about, order)
#
#    @abstractmethod
#    def _series_expand(self, param, about, order):
#        raise NotImplementedError(self.__class__.__name__)
#
#
#    def __add__(self, other):
#        if isinstance(other, SuperOperator.scalar_types):
#            return SuperOperatorPlus.create(self, other * IdentitySuperOperator)
#        elif isinstance(other, SuperOperator):
#            return SuperOperatorPlus.create(self, other)
#        return NotImplemented
#
#    __radd__ = __add__
#
#    def __mul__(self, other):
#        if isinstance(other, SuperOperator.scalar_types):
#            return ScalarTimesSuperOperator.create(other, self)
#        elif isinstance(other, Operator):
#            return SuperOperatorTimesOperator.create(self, other)
#        elif isinstance(other, SuperOperator):
#            return SuperOperatorTimes.create(self, other)
#        return NotImplemented
#
#    def __rmul__(self, other):
#        if isinstance(other, SuperOperator.scalar_types):
#            return ScalarTimesSuperOperator.create(other, self)
#        return NotImplemented
#
#    def __sub__(self, other):
#        return self + (-1) * other
#
#    def __rsub__(self, other):
#        return (-1) * self + other
#
#    def __neg__(self):
#        return (-1) * self
#
#    def __div__(self, other):
#        if isinstance(other, SuperOperator.scalar_types):
#            return self * (sympyOne / other)
#        return NotImplemented
#
#
#@check_signature
#class SuperOperatorSymbol(SuperOperator, Operation):
#    """
#    Operator Symbol class, parametrized by an identifier string and an associated Hilbert space.
#        OperatorSymbol(name, hs)
#    :param name: Symbol identifier
#    :type name: str
#    :param hs: Associated Hilbert space.
#    :type hs: HilbertSpace
#    """
#    signature = str, (HilbertSpace, str, int, tuple)
#
#    def __init__(self, name, hs):
#        if isinstance(hs, (str, int)):
#            hs = local_space(hs)
#        elif isinstance(hs, tuple):
#            hs = prod([local_space(h) for h in hs], neutral=TrivialSpace)
#        super(OperatorSymbol, self).__init__(name, hs)
#
#    def __str__(self):
#        return self.operands[0]
#
#    def _tex(self):
#        return r"\hat{{{}}}".format(identifier_to_tex(self.operands[0]))
#
#    def _to_qutip(self, full_space=None):
#        raise AlgebraError("Cannot convert super operator symbol to representation matrix. Substitute first.")
#
#    @property
#    def _space(self):
#        return self.operands[1]
#
#    def _expand(self):
#        return self
#
#    def _series_expand(self, param, about, order):
#        return (self,) + (() * order)
#
#
#@singleton
#class IdentitySuperOperator(SuperOperator, Expression):
#    """
#    IdentitySuperOperator constant (singleton) object.
#    """
#
#    @property
#    def _space(self):
#        return TrivialSpace
#
#    def _superadjoint(self):
#        return self
#
#    def _to_qutip(self, full_space):
#        return qutip.spre(qutip.tensor(*[qutip.qeye(s.dimension) for s in full_space.local_factors()]))
#
##    def mathematica(self):
##        return "IdentityOperator"
#
#    def _expand(self):
#        return self
#
#    def _series_expand(self, param, about, order):
#        return (self,) + (() * order)
#
##    def _pseudo_inverse(self):
##        return self
#
#    def _tex(self):
#        return r"\hat{1}"
#
#    def __str__(self):
#        return "_1_"
#
#    def __eq__(self, other):
#        return self is other or other == 1
#
#
#
#
#
#@singleton
#class SuperOperatorZero(SuperOperator, Expression):
#    """
#    ZeroSuperOperator constant (singleton) object.
#    """
#
#    @property
#    def _space(self):
#        return TrivialSpace
#
#    def _superadjoint(self):
#        return self
#
#    def _to_qutip(self, full_space):
#        return qutip.spre(ZeroOperator.to_qutip(full_space))
#
#    def _expand(self):
#        return self
#
#    def _series_expand(self, param, about, order):
#        return (self,) + (()*order)
#
##    def _pseudo_inverse(self):
##        return self
#
#    def _tex(self):
#        return r"\hat{0}"
#
#    def __eq__(self, other):
#        return self is other or other == 0
#
#    def __str__(self):
#        return "_0_"
#
#
#class SuperOperatorOperation(SuperOperator, Operation):
#    """
#    Base class for Operations acting only on SuperOperator arguments.
#    """
#    signature = SuperOperator,
#
#    @property
#    def _space(self):
#        return prod((o.space for o in self.operands), TrivialSpace)
#
#
#
#
#@assoc
#@orderby
#@filter_neutral
#@match_replace_binary
#@filter_neutral
#@check_signature_assoc
#class SuperOperatorPlus(SuperOperatorOperation):
#    """
#    A sum of Operators
#        OperatorPlus(*summands)
#    :param summands: Operator summands.
#    :type summands: Operator
#    """
#    neutral_element = SuperOperatorZero
#    _binary_rules = []
#
#    @classmethod
#    def order_key(cls, a):
#        if isinstance(a, ScalarTimesSuperOperator):
#            return Operation.order_key(a.term), a.coeff
#        return Operation.order_key(a), 1
#
#    def _to_qutip(self, full_space=None):
#        if full_space == None:
#            full_space = self.space
#        assert self.space <= full_space
#        return sum((op.to_qutip(full_space) for op in self.operands), 0)
#
#    def _expand(self):
#        return sum((o.expand() for o in self.operands), ZeroOperator)
#
#    def _series_expand(self, param, about, order):
#        res = tuple_sum((o.series_expand(param, about, order) for o in self.operands), ZeroOperator)
#        return res
#
#    def _tex(self):
#        ret = self.operands[0].tex()
#
#        for o in self.operands[1:]:
#            if isinstance(o, ScalarTimesOperator) and ScalarTimesOperator.has_minus_prefactor(o.coeff):
#                ret += " - " + tex(-o)
#            else:
#                ret += " + " + tex(o)
#        return ret
#
#    def __str__(self):
#        ret = str(self.operands[0])
#
#        for o in self.operands[1:]:
#            if isinstance(o, ScalarTimesOperator) and ScalarTimesOperator.has_minus_prefactor(o.coeff):
#                ret += " - " + str(-o)
#            else:
#                ret += " + " + str(o)
#        return ret
#
#
#
#@assoc
#@orderby
#@filter_neutral
#@match_replace_binary
#@filter_neutral
#@check_signature_assoc
#class OperatorTimes(OperatorOperation):
#    """
#    A product of Operators that serves both as a product within a Hilbert space as well as a tensor product.
#        OperatorTimes(*factors)
#    :param factors: Operator factors.
#    :type factors: Operator
#    """
#
#    neutral_element = IdentityOperator
#    _binary_rules = []
#
#    class OperatorOrderKey(object):
#        """
#        Auxiliary class that generates the correct pseudo-order relation for operator products.
#        Only operators acting on different Hilbert spaces are commuted to achieve the order specified in the full HilbertSpace.
#        I.e., sorted(factors, key = OperatorOrderKey) achieves this ordering.
#        """
#        def __init__(self, op):
#            space = op.space
#            self.op = op
#            self.full = False
#            self.trivial = False
#            if isinstance(space, LocalSpace):
#                self.local_spaces = {space.operands, }
#            elif space is TrivialSpace:
#                self.local_spaces = set(())
#                self.trivial = True
#            elif space is FullSpace:
#                self.full = True
#            else:
#                assert isinstance(space, ProductSpace)
#                self.local_spaces = {s.operands for s in space.operands}
#
#        def __lt__(self, other):
#            if self.trivial and other.trivial:
#                return Operation.order_key(self.op) < Operation.order_key(other.op)
#
#            if self.full or len(self.local_spaces & other.local_spaces):
#                return False
#            return tuple(self.local_spaces) < tuple(other.local_spaces)
#
#        def __gt__(self, other):
#            if self.trivial and other.trivial:
#                return Operation.order_key(self.op) > Operation.order_key(other.op)
#
#            if self.full or len(self.local_spaces & other.local_spaces):
#                return False
#
#            return tuple(self.local_spaces) > tuple(other.local_spaces)
#
#        def __eq__(self, other):
#            if self.trivial and other.trivial:
#                return Operation.order_key(self.op) == Operation.order_key(other.op)
#
#            return self.full or len(self.local_spaces & other.local_spaces) > 0
#
#
#    order_key = OperatorOrderKey
#
#    @classmethod
#    def create(cls, *ops):
#        if any(o == ZeroOperator for o in ops):
#            return ZeroOperator
#        return cls(*ops)
#
#    def factor_for_space(self, spc):
#        if spc == TrivialSpace:
#            ops_on_spc = [o for o in self.operands if o.space is TrivialSpace]
#            ops_not_on_spc = [o for o in self.operands if o.space > TrivialSpace]
#        else:
#            ops_on_spc = [o for o in self.operands if (o.space & spc) > TrivialSpace]
#            ops_not_on_spc = [o for o in self.operands if (o.space & spc) is TrivialSpace]
#        return OperatorTimes.create(*ops_on_spc),OperatorTimes.create(*ops_not_on_spc)
#
#
#    def _to_qutip(self, full_space=None):
#
#        # if any factor acts non-locally, we need to expand distributively.
#        if any(len(op.space) > 1 for op in self.operands):
#            se = self.expand()
#            if se == self:
#                raise ValueError("Cannot represent as QuTiP object: {!s}".format(self))
#            return se.to_qutip(full_space)
#
#
#        if full_space == None:
#            full_space = self.space
#
#        all_spaces = full_space.local_factors()
#        by_space = []
#        ck = 0
#        for ls in all_spaces:
#            # group factors by associated local space
#            ls_ops = [o.to_qutip() for o in self.operands if o.space == ls]
#            if len(ls_ops):
#                # compute factor associated with local space
#                by_space.append(prod(ls_ops))
#                ck += len(ls_ops)
#            else:
#                # if trivial action, take identity matrix
#                by_space.append(qutip.qeye(ls.dimension))
#        assert ck == len(self.operands)
#        # combine local factors in tensor product
#        return qutip.tensor(*by_space)
#
#    def _expand(self):
#        eops = [o.expand() for o in self.operands]
#        # store tuples of summands of all expanded factors
#        eopssummands = [eo.operands if isinstance(eo, OperatorPlus) else (eo,) for eo in eops]
#        # iterate over a cartesian product of all factor summands, form product of each tuple and sum over result
#        return sum((OperatorTimes.create(*combo) for combo in cartesian_product(*eopssummands)), ZeroOperator)
#
#    def _tex(self):
#        ret = self.operands[0].tex()
#        for o in self.operands[1:]:
#            if isinstance(o, OperatorPlus):
#                ret += r" \left({}\right) ".format(tex(o))
#            else:
#                ret += " {}".format(tex(o))
#        return ret
#
#    def __str__(self):
#        ret = str(self.operands[0])
#        for o in self.operands[1:]:
#            if isinstance(o, OperatorPlus):
#                ret += r" ({})".format(str(o))
#            else:
#                ret += " {}".format(str(o))
#        return ret
#
#
#@match_replace
#@check_signature
#class ScalarTimesOperator(Operator, Operation):
#    """
#    Multiply an operator by a scalar coefficient.
#        ScalarTimesOperator(coefficient, term)
#    :param coefficient: Scalar coefficient.
#    :type coefficient: Any of Operator.scalar_types
#    :param term: The operator that is multiplied.
#    :type term: Operator
#    """
#    signature = Operator.scalar_types, Operator
#    _rules = []
#
#    @staticmethod
#    def has_minus_prefactor(c):
#        """
#        For a scalar object c, determine whether it is prepended by a "-" sign.
#        """
#        cs = str(c).strip()
#        return cs[0] == "-"
#
#
#
#    @property
#    def _space(self):
#        return self.operands[1].space
#
#    @property
#    def coeff(self):
#        return self.operands[0]
#
#    @property
#    def term(self):
#        return self.operands[1]
#
#
#    def _tex(self):
#        coeff, term = self.operands
#
#        if isinstance(coeff, Add):
#            cs = r" \left({}\right)".format(tex(coeff))
#        else:
#            cs = " {}".format(tex(coeff))
#
#        if term == IdentityOperator:
#            ct = ""
#        if isinstance(term, OperatorPlus):
#            ct = r" \left({}\right)".format(term.tex())
#        else:
#            ct = r" {}".format(term.tex())
#
#        return cs + ct
#
#    def __str__(self):
#        coeff, term = self.operands
#
#        if isinstance(coeff, Add):
#            cs = r"({!s})".format()
#        else:
#            cs = " {!s}".format(coeff)
#
#        if term == IdentityOperator:
#            ct = ""
#        if isinstance(term, OperatorPlus):
#            ct = r" ({!s})".format(term)
#        else:
#            ct = r" {!s}".format(term)
#
#        return cs + ct
#
#
#    def _to_qutip(self, full_space=None):
#        return complex(self.coeff) * self.term.to_qutip(full_space)
#
#    def _expand(self):
#        c, t = self.operands
#        et = t.expand()
#        if isinstance(et, OperatorPlus):
#            return sum((c * eto for eto in et.operands), ZeroOperator)
#        return c * et
#
#    def _pseudo_inverse(self):
#        c, t = self.operands
#        return t.pseudo_inverse() / c
#
#
#    def __complex__(self):
#        if self.term is IdentityOperator:
#            return complex(self.coeff)
#        return NotImplemented
#
#    def __float__(self):
#        if self.term is IdentityOperator:
#            return float(self.coeff)
#        return NotImplemented
#
#
#
#
#def safe_tex(obj):
#    if isinstance(obj, (int, float, complex)):
#        return format_number_for_tex(obj)
#
#    if isinstance(obj, SympyBasic):
#        return sympy_latex(obj).strip('$')
#    try:
#        return obj.tex()
#    except AttributeError:
#        return r"{\rm " + str(obj) + "}"
#
#import abstract_algebra
#tex = abstract_algebra.tex = safe_tex
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
##
##def format_number_for_mathematica(num):
##    if num == 0: #also True for 0., 0j
##        return "0"
##    if isinstance(num, complex):
##        if num.imag == 0:
##            return format_number_for_tex(num.real)
##        return "Complex[%g,%g]" % (num.real, num.imag)
##
##    return "%g" % num
##
#
#
#
#greek_letter_strings = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
#                        "zeta", "eta", "theta", "vartheta", "iota", "kappa",
#                        "lambda", "mu", "nu", "xi", "pi", "varpi", "rho",
#                        "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
#                        "varphi", "chi", "psi", "omega",
#                        "Gamma", "Delta", "Theta", "Lambda", "Xi",
#                        "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]
#greekToLatex = {"alpha":"Alpha", "beta":"Beta", "gamma":"Gamma", "delta":"Delta", "epsilon":"Epsilon", "varepsilon":"Epsilon",
#                        "zeta":"Zeta", "eta":"Eta", "theta":"Theta", "vartheta":"Theta", "iota":"Iota", "kappa":"Kappa",
#                        "lambda":"Lambda", "mu":"Mu", "nu":"Nu", "xi":"Xi", "pi":"Pi", "varpi":"Pi", "rho":"Rho",
#                        "varrho":"Rho", "sigma":"Sigma", "varsigma":"Sigma", "tau":"Tau", "upsilon":"Upsilon", "phi": "Phi",
#                        "varphi":"Phi", "chi":"Chi", "psi":"Psi", "omega":"Omega",
#                        "Gamma":"CapitalGamma", "Delta":"CapitalDelta", "Theta":"CapitalTheta", "Lambda":"CapitalLambda", "Xi":"CapitalXi",
#                        "Pi":"CapitalPi", "Sigma":"CapitalSigma", "Upsilon":"CapitalUpsilon", "Phi":"CapitalPhi", "Psi":"CapitalPsi", "Omega":"CapitalOmega"
#                }
#
#import re
#_idtp = re.compile(r'(?!\\)({})(\b|_)'.format("|".join(greek_letter_strings)))
#
#def identifier_to_tex(identifier):
#    """
#    If an identifier contains a greek symbol name as a separate word,
#    (e.g. 'my_alpha_1' contains 'alpha' as a separate word, but 'alphaman' doesn't)
#    add a backslash in front.
#    """
##    identifier = creduce(lambda a,b: "{%s_%s}" % (b, a), ["{%s}" % part for part in reversed(identifier.split("_"))])
#
#    return _idtp.sub(r'{\\\1}\2', identifier)
#
#
#
#@check_signature
#@match_replace
#class Adjoint(OperatorOperation):
#    """
#    The symbolic Adjoint of an operator.
#        Adjoint(op)
#    :param op: The operator to take the adjoint of.
#    :type op: Operator
#    """
#    @property
#    def operand(self):
#        return self.operands[0]
#
#    _rules = []
#
#    def _to_qutip(self, full_space=None):
#        return qutip.dag(self.operands[0].to_qutip(full_space))
#
#    def _expand(self):
#        eo = self.operand.expand()
#        if isinstance(eo, OperatorPlus):
#            return sum((eoo.adjoint() for eoo in eo.operands), ZeroOperator)
#        return eo._adjoint()
#
#    def _pseudo_inverse(self):
#        return self.operand.pseudo_inverse().adjoint()
#
#    def _tex(self):
#        return "\left(" + self.operands[0].tex() + r"\right)^\dagger"
#
#    def __str__(self):
#        if isinstance(self.operand, OperatorSymbol):
#            return "{}^*".format(str(self.operand))
#        return "({})^*".format(str(self.operand))
#
## for hilbert space dimensions less than or equal to this,
## compute numerically PseudoInverse and NullSpaceProjector representations
#DENSE_DIMENSION_LIMIT = 1000
#
#@check_signature
#@match_replace
#class PseudoInverse(OperatorOperation):
#    """
#    The symbolic pseudo-inverse of an operator.
#        PseudoInverse(op)
#    :param op: The operator to take the adjoint of.
#    :type op: Operator
#    """
#    delegate_to_method = ScalarTimesOperator, Squeeze, Displace, ZeroOperator.__class__, IdentityOperator.__class__
#
#    @classmethod
#    def create(cls, op):
#        if isinstance(op, cls.delegate_to_method):
#            return op._pseudo_inverse()
#        return super(PseudoInverse, cls).create(op)
#
#    @property
#    def operand(self):
#        return self.operands[0]
#
#    _rules = []
#
#    def _to_qutip(self, full_space=None):
#        mo = self.operand.to_qutip(full_space)
#        if full_space.dimension <= DENSE_DIMENSION_LIMIT:
#            arr = mo.data.toarray()
#            from scipy.linalg import pinv
#            piarr = pinv(arr)
#            pimo = qutip.Qobj(piarr)
#            pimo.dims = mo.dims
#            pimo.isherm = mo.isherm
#            pimo.type = 'oper'
#            return pimo
#        raise NotImplementedError("Only implemented for smaller state spaces")
##        return qutip.dag(self.operands[0].to_qutip(full_space))
#
#    def _expand(self):
#        return self
#
#    def _pseudo_inverse(self):
#        return self.operand
#
#    def _tex(self):
#        return "\left(" + self.operands[0].tex() + r"\right)^+"
#
#    def __str__(self):
#        if isinstance(self.operand, OperatorSymbol):
#            return "{}^+".format(str(self.operand))
#        return "({})^+".format(str(self.operand))
#
#PseudoInverse.delegate_to_method = PseudoInverse.delegate_to_method + (PseudoInverse,)
#
#
#@check_signature
#@match_replace
#class NullSpaceProjector(OperatorOperation):
#    """
#    Returns a projection operator that projects onto the nullspace of its operand
#        NullSpaceProjector(op)
#    I.e. `op * NullSpaceProjector(op) == 0`
#    :param op: Operator argument
#    :type op: Operator
#    """
#
#    _rules = []
#
#    @property
#    def operand(self):
#        return self.operands[0]
#
#    def to_qutip(self, full_space=None):
#        mo = self.operand.to_qutip(full_space)
#        if full_space.dimension <= DENSE_DIMENSION_LIMIT:
#            arr = mo.data.toarray()
#            from scipy.linalg import svd
#            # compute Singular Value Decomposition
#            U, s, Vh = svd(arr)
#            tol = 1e-8 * s[0]
#            zero_svs = s < tol
#            Vhzero = Vh[zero_svs,:]
#            PKarr = Vhzero.conjugate().transpose().dot(Vhzero)
#            PKmo = qutip.Qobj(PKarr)
#            PKmo.dims = mo.dims
#            PKmo.isherm = True
#            PKmo.type = 'oper'
#            return PKmo
#        raise NotImplementedError("Only implemented for smaller state spaces")
#
#
#    def _tex(self):
#        return r"\mathcal{P}_{{\rm Ker}" + tex(self.operand) + "}"
#
#    def __str__(self):
#        return "P_ker({})".format(str(self.operand))
#
#
#
#
#@implied_local_space
#@match_replace
#@check_signature
#class OperatorTrace(Operator, Operation):
#    """
#    Take the (partial) trace of an operator `op` over the degrees of freedom given by `space`.
#        OperatorTrace(space, op)
#    :param space: The degrees of freedom to trace over
#    :type space: HilbertSpace
#    :param op: The operator to take the trace of.
#    :type op: Operator
#    """
#    signature = HilbertSpace, Operator
#    _rules = []
#
#    def __init__(self, space, op):
#        if isinstance(space, (int, str)):
#            space = local_space(space)
#        super(OperatorTrace, self).__init__(space, op)
#
#    @property
#    def _space(self):
#        over_space, op = self.operands
#        return op.space / over_space
#
#    def _expand(self):
#        s, o = self.operands
#        return OperatorTrace.create(s, o.expand())
#
#    def _tex(self):
#        s, o = self.operands
#        return r"{{\rm Tr}}_{{{}}} \left[ {} \right]".format(tex(s), tex(o))
#
#    def _tex(self):
#        s, o = self.operands
#        return r"tr_{!s}[{!s}]".format(s, o)
#
#tr = OperatorTrace.create
#
#
### Expression rewriting _rules
#u = wc("u", head=Operator.scalar_types)
#v = wc("v", head=Operator.scalar_types)
#
#n = wc("n", head=(int, str))
#m = wc("m", head=(int, str))
#
#
#A = wc("A", head=Operator)
#A__ = wc("A__", head=Operator)
#A___ = wc("A___", head=Operator)
#B = wc("B", head=Operator)
#B__ = wc("B__", head=Operator)
#B___ = wc("B___", head=Operator)
#
#
#A_plus = wc("A", head=OperatorPlus)
#A_times = wc("A", head=OperatorTimes)
#A_local = wc("A", head = LocalOperator)
#B_local = wc("B", head = LocalOperator)
#
#ls = wc("ls", head=LocalSpace)
#h1 = wc("h1", head = HilbertSpace)
#h2 = wc("h2", head = HilbertSpace)
#H_ProductSpace = wc("H", head = ProductSpace)
#
#ra = wc("ra", head=(int, str))
#rb = wc("rb", head=(int, str))
#rc = wc("rc", head=(int, str))
#rd = wc("rd", head=(int, str))
#
#ScalarTimesOperator._rules += [
#    ((1, A), lambda A: A),
#    ((0, A), lambda A: ZeroOperator),
#    ((u, ZeroOperator), lambda u: ZeroOperator),
#    ((u, ScalarTimesOperator(v, A)), lambda u, v, A: (u * v) * A)
#]
#
#OperatorPlus._binary_rules += [
#    ((ScalarTimesOperator(u, A), ScalarTimesOperator(v, A)), lambda u, v, A: (u + v) * A),
#    ((ScalarTimesOperator(u, A), A), lambda u, A: (u + 1) * A),
#    ((A, ScalarTimesOperator(v, A)), lambda v, A: (1 + v) * A),
#    ((A, A), lambda A: 2 * A),
#]
#
#OperatorTimes._binary_rules += [
#    ((ScalarTimesOperator(u, A), B), lambda u, A, B: u * (A * B)),
#
#    ((A, ScalarTimesOperator(u, B)), lambda A, u, B: u * (A * B)),
#
#    ((LocalSigma(ls, ra, rb), LocalSigma(ls, rc, rd)),
#     lambda ls, ra, rb, rc, rd: LocalSigma(ls, ra, rd)
#                                    if rb == rc else ZeroOperator),
#
#    ((Create(ls), LocalSigma(ls, rc, rd)),
#     lambda ls, rc, rd: sqrt(rc + 1) * LocalSigma(ls, rc + 1, rd)),
#
#    ((Destroy(ls), LocalSigma(ls, rc, rd)),
#     lambda ls, rc, rd: sqrt(rc) * LocalSigma(ls, rc - 1, rd)),
#
#    ((LocalSigma(ls, rc, rd), Destroy(ls)),
#     lambda ls, rc, rd: sqrt(rd + 1) * LocalSigma(ls, rc, rd + 1)),
#
#    ((LocalSigma(ls, rc, rd), Create(ls)),
#     lambda ls, rc, rd: sqrt(rd) * LocalSigma(ls, rc, rd - 1)),
#
#    ((Destroy(ls), Create(ls)),
#     lambda ls: IdentityOperator + Create(ls) * Destroy(ls)),
#
#    ((Phase(ls, u), Phase(ls, v)), lambda ls, u, v: Phase.create(ls, u + v)),
#    ((Displace(ls, u), Displace(ls, v)), lambda ls, u, v: exp((u*v.conjugate() - u.conjugate() * v)/2) * Displace.create(ls, u + v)),
#
#    ((Destroy(ls), Phase(ls, u)), lambda ls, u: exp(I*u) * Phase(ls, u) * Destroy(ls)),
#    ((Destroy(ls), Displace(ls, u)), lambda ls, u: Displace(ls, u) * (Destroy(ls) + u)),
#
#    ((Phase(ls, u), Create(ls)), lambda ls, u: exp(I*u) * Create(ls) * Phase(ls, u)),
#    ((Displace(ls, u), Create(ls)), lambda ls, u: (Create(ls) - u.conjugate())* Displace(ls, u)),
#
#    ((Phase(ls, u), LocalSigma(ls, n, m)), lambda ls, u, n, m: exp(I* u * n) * LocalSigma(ls, n, m)),
#    ((LocalSigma(ls, n, m), Phase(ls, u)), lambda ls, u, n, m: exp(I* u * m) * LocalSigma(ls, n, m)),
#]
#
#Adjoint._rules += [
#    ((ScalarTimesOperator(u, A),), lambda u, A: u.conjugate() * A.adjoint()),
#    ((A_plus,), lambda A: OperatorPlus.create(*[o.adjoint() for o in A.operands])),
#    ((A_times,), lambda A: OperatorTimes.create(*[o.adjoint() for o in A.operands[::-1]])),
#    ((Adjoint(A),), lambda A: A),
#    ((Create(ls),), lambda ls: Destroy(ls)),
#    ((Destroy(ls),), lambda ls: Create(ls)),
#    ((LocalSigma(ls, ra, rb),), lambda ls, ra, rb: LocalSigma(ls, rb, ra)),
#]
#
#Displace._rules +=[
#    ((ls, 0), lambda ls: IdentityOperator)
#]
#Phase._rules +=[
#    ((ls, 0), lambda ls: IdentityOperator)
#]
#Squeeze._rules +=[
#    ((ls, 0), lambda ls: IdentityOperator)
#]
#
#
#def factor_for_trace(ls, op):
#    """
#    Given a local space ls to take the partial trace over and an operator, factor the trace such that operators acting on
#    disjoint degrees of freedom are pulled out of the trace. If the operator acts trivially on ls the trace yields only
#    a pre-factor equal to the dimension of ls. If there are LocalSigma operators among a product, the trace's cyclical property
#    is used to move to sandwich the full product by LocalSigma operators:
#        Tr A sigma_jk B = Tr sigma_jk B A sigma_jj
#    :param ls: Degree of Freedom to trace over
#    :type ls: HilbertSpace
#    :param op: Operator to take the trace of
#    :type op: Operator
#    :return: The (partial) trace over the operator's spc-degrees of freedom
#    :rtype: Operator
#    """
#    if op.space == ls:
#        if isinstance(op, OperatorTimes):
#            pull_out = [o for o in op.operands if o.space is TrivialSpace]
#            rest = [o for o in op.operands if o.space is not TrivialSpace]
#            if pull_out:
#                return OperatorTimes.create(*pull_out) * OperatorTrace.create(ls, OperatorTimes.create(*rest))
#        raise CannotSimplify()
#    if ls & op.space == TrivialSpace:
#        return ls.dimension * op
#    if ls < op.space and isinstance(op, OperatorTimes):
#
#        pull_out = [o for o in op.operands if (o.space & ls) == TrivialSpace]
#
#        rest = [o for o in op.operands if (o.space & ls) != TrivialSpace]
#        if not isinstance(rest[0], LocalSigma) or not isinstance(rest[-1], LocalSigma):
#            found_ls = False
#            for j, r in enumerate(rest):
#                if isinstance(r, LocalSigma):
#                    found_ls = True
#                    break
#            if found_ls:
#                m, n = r.operands[1:]
#                rest = rest[j:] + rest[:j] + [LocalSigma(ls, m, m)]
#        if not rest:
#            rest = [IdentityOperator]
#        if len(pull_out):
#            return OperatorTimes.create(*pull_out) * OperatorTrace.create(ls, OperatorTimes.create(*rest))
#    raise CannotSimplify()
#
#
#def decompose_space(H, A):
#    return OperatorTrace.create(ProductSpace.create(*H.operands[:-1]),
#        OperatorTrace.create(H.operands[-1], A))
#
#OperatorTrace._rules += [
#    ((TrivialSpace, A), lambda A: A),
#    ((h1, ZeroOperator), lambda h1: ZeroOperator),
#    ((h1, IdentityOperator), lambda h1: h1.dimension * IdentityOperator),
#    ((h1, A_plus), lambda h1, A: sum(OperatorTrace.create(h1, o) for o in A.operands)),
#    ((h1, Adjoint(A)), lambda h1, A: Adjoint.create(OperatorTrace.create(h1, A))),
#    ((h1, ScalarTimesOperator(u, A)), lambda h1, u, A: u * OperatorTrace.create(h1, A)),
#    ((H_ProductSpace, A), lambda H, A : decompose_space(H, A)),
#    ((ls, Create(ls)), lambda ls: ZeroOperator),
#    ((ls, Destroy(ls)), lambda ls: ZeroOperator),
#    ((ls, LocalSigma(ls, n, m)), lambda ls, n, m: IdentityOperator if n == m else ZeroOperator),
#    ((ls, A), lambda ls, A: factor_for_trace(ls, A)),
#]
#
#
#
#
#class NonSquareMatrix(Exception):
#    pass
#
#
#class Matrix(Expression):
#    """
#    Matrix with Operator (or scalar-) valued elements.
#
#    """
#    matrix = None
#    _hash = None
#
#    def __init__(self, m):
#        if isinstance(m, ndarray):
#            self.matrix = m
#        elif isinstance(m, Matrix):
#            self.matrix = np_array(m.matrix)
#        else:
#            self.matrix = np_array(m)
#        if len(self.matrix.shape) < 2:
#            self.matrix = self.matrix.reshape((1, self.matrix.shape[0]))
#        if len(self.matrix.shape) > 2:
#            raise ValueError()
#
#
#    @property
#    def shape(self):
#        return self.matrix.shape
#
#    def __hash__(self):
#        if not self._hash:
#            self._hash = hash((tuple(self.matrix.flatten()), self.matrix.shape, Matrix))
#        return self._hash
#
#    def __eq__(self, other):
#        return isinstance(other, Matrix) and (self.matrix == other.matrix).all()
#
#    def __add__(self, other):
#        if isinstance(other, Matrix):
#            return Matrix(self.matrix + other.matrix)
#        else: return Matrix(self.matrix + other)
#
#    def __radd__(self, other):
#        return Matrix(other + self.matrix)
#
#    def __mul__(self, other):
#        if isinstance(other, Matrix):
#            return Matrix(self.matrix.dot(other.matrix))
#        else: return Matrix(self.matrix * other)
#
#    def __rmul__(self, other):
#        return Matrix(other * self.matrix)
#
#    def __sub__(self, other):
#        return self + (-1) * other
#
#    def __rsub__(self, other):
#        return (-1) * self + other
#
#    def __neg__(self):
#        return (-1) * self
#
#
#    #    @_trace
#    def __div__(self, other):
#        if isinstance(other, Operator.scalar_types):
#            return self * (sympyOne / other)
#        return NotImplemented
#
#    __truediv__ = __div__
#
#    #    def __pow__(self, power):
#    #        return OperatorMatrix(self.matrix.__pow__(power))
#
#    def transpose(self):
#        """
#        :return: The transpose matrix
#        :rtype: Matrix
#        """
#        return Matrix(self.matrix.T)
#
#    def conjugate(self):
#        """
#        The element-wise conjugate matrix, i.e., if an element is an operator this means the adjoint operator,
#        but no transposition of matrix elements takes place.
#        :return: Element-wise hermitian conjugate matrix.
#        :rtype: Matrix
#        """
#        return Matrix(np_conjugate(self.matrix))
#
#    @property
#    def T(self):
#        """
#        :return: Transpose matrix
#        :rtype: Matrix
#        """
#        return self.transpose()
#
#    def adjoint(self):
#        """
#        Return the adjoint operator matrix, i.e. transpose and the Hermitian adjoint operators of all elements.
#        """
#        return self.T.conjugate()
#
#    dag = adjoint
#
#    def __repr__(self):
#        return "OperatorMatrix({})".format(repr(self.matrix.tolist()))
#
#    def trace(self):
#        if self.shape[0] == self.shape[1]:
#            return sum(self.matrix[k, k] for k in range(self.shape[0]))
#        raise NonSquareMatrix(repr(self))
#
#
#    @property
#    def H(self):
#        """
#        Return the adjoint operator matrix, i.e. transpose and the Hermitian adjoint operators of all elements.
#        """
#        return self.adjoint()
#
#
#    def __getitem__(self, item_id):
#        item = self.matrix.__getitem__(item_id)
#        if isinstance(item, ndarray):
#            return Matrix(item)
#        return item
#
#    def element_wise(self, method):
#        """
#        Apply a method to each matrix element and return the result in a new operator matrix of the same shape.
#        :param method: A method taking a single argument.
#        :type method: FunctionType
#        :return: Operator matrix with results of method applied element-wise.
#        :rtype: Matrix
#        """
#        s = self.shape
#        emat = [method(o) for o in self.matrix.flatten()]
#        return Matrix(np_array(emat).reshape(s))
#
#
#    def expand(self):
#        """
#        Expand each matrix element distributively.
#        :return: Expanded matrix.
#        :rtype: Matrix
#        """
#        m = lambda o: o.expand() if isinstance(o, Operator) else o
#        return self.element_wise(m)
#
#    def _substitute(self, var_map):
#        m = lambda o: substitute(o, var_map) if isinstance(o, Operation) else o
#        return self.element_wise(m)
#
#    def _all_symbols(self):
#        return set_union()
#
#
#    def _tex(self):
#        ret = r"\begin{pmatrix} "
##        for row in self.matrix:
#        ret += r""" \\
#""".join([" & ".join([tex(o) for o in row]) for row in self.matrix])
#        ret += r"\end{pmatrix}"
#
#        return ret
#
#
#
#
#    @property
#    def space(self):
#        """
#        :return: Return the combined Hilbert space of all matrix elements.
#        :rtype: HilbertSpace
#        """
#        return prod((space(o) for o in self.matrix.flatten()), TrivialSpace)
#
#
#def hstack(matrices):
#    """
#    Generalizes `numpy.hstack` to OperatorMatrix objects.
#    """
#    return Matrix(np_hstack(matrices))
#
#
#def vstack(matrices):
#    """
#    Generalizes `numpy.vstack` to OperatorMatrix objects.
#    """
#    return Matrix(np_vstack(matrices))
#
#
#def diag(v, k=0):
#    """
#    Generalizes the diagonal matrix creation capabilities of `numpy.diag` to OperatorMatrix objects.
#    """
#    return Matrix(np_diag(v, k))
#
#
#def block_matrix(A, B, C, D):
#    """
#    Generate the operator matrix with quadrants
#       [[A B]
#        [C D]]
#    :return: The combined block matrix [[A, B], [C, D]].
#    :type: OperatorMatrix
#    """
#    return vstack((hstack((A, B)), hstack((C, D))))
#
#
#def identity_matrix(N):
#    """
#    Generate the N-dimensional identity matrix.
#    :param N: Dimension
#    :type N: int
#    :return: Identity matrix in N dimensions
#    :rtype: Matrix
#    """
#    return diag(np_ones(N, dtype=int))
#
#
#def zeros(shape):
#    """
#    Generalizes `numpy.zeros` to OperatorMatrix objects.
#    """
#    return Matrix(np_zeros(shape))
#
#
#def permutation_matrix(permutation):
#    """
#    Return an orthogonal permutation matrix M_sigma
#    for a permutation sigma given by a tuple
#    (sigma(1), sigma(2),... sigma(n)), such that
#    such that M_sigma e_i = e_sigma(i), where e_k
#    is the k-th standard basis vector.
#    This definition ensures a composition law:
#    M_{sigma . pi} = M_sigma M_pi.
#    In column form M_sigma is thus given by
#    M = [e_sigma(1), e_sigma(2), ... e_sigma(n)].
#    """
#    assert(check_permutation(permutation))
#    n = len(permutation)
#    op_matrix = zeros((n, n))
#    for i, j in enumerate(permutation):
#        op_matrix[j, i] = 1
#    return op_matrix
#
## :deprecated:
## for backwards compatibility
#OperatorMatrixInstance = Matrix
#IdentityMatrix = identity_matrix
#
#def Im(op):
#    """
#    The imaginary part of a number or operator. Acting on OperatorMatrices, it produces the element-wise imaginary parts.
#    :param op: Anything that has a conjugate method.
#    :type op: Operator or Matrix or any of Operator.scalar_types
#    :return: The imaginary part of the operand.
#    :rtype: Same as type of `op`.
#    """
#    return (op.conjugate() - op) * I / 2
#
#def Re(op):
#    """
#    The real part of a number or operator. Acting on OperatorMatrices, it produces the element-wise real parts.
#    :param op: Anything that has a conjugate method.
#    :type op: Operator or Matrix or any of Operator.scalar_types
#    :return: The real part of the operand.
#    :rtype: Same as type of `op`.
#    """
#    return (op.conjugate()+ op) / 2
#
#
#def ImAdjoint(opmatrix):
#    """
#    The imaginary part of an OperatorMatrix, i.e. a hermitian OperatorMatrix
#    :param opmatrix: The operand.
#    :type opmatrix: Matrix
#    :return: The matrix imaginary part of the operand.
#    :rtype: Matrix
#    """
#    return (opmatrix.H - opmatrix) * I / 2
#
#
#def ReAdjoint(opmatrix):
#    """
#    The real part of an OperatorMatrix, i.e. a hermitian OperatorMatrix
#    :param opmatrix: The operand.
#    :type opmatrix: Matrix
#    :return: The matrix real part of the operand.
#    :rtype: Matrix
#    """
#    return (opmatrix.H + opmatrix) / 2
#
#
#
#
#
#
