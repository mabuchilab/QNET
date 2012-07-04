from __future__ import division
from abstract_algebra import *
from hilbert_space_algebra import *
from helpers import *
from sympy import exp, log, cos, sin, cosh, sinh, tan, cot, \
                            acos, asin, acosh, asinh, atan, atan2, atanh, acot, sqrt, \
                            factorial, pi, I, sympify, Basic as SympyBasic

sympyOne = sympify(1)

import qutip




class Operator(object):
    """
    The basic operator class, which fixes the abstract interface of operator objects 
    and where possible also defines the default behavior under operations.
    Any operator contains an associated HilbertSpace object, 
    on which it is taken to act non-trivially.
    """

    scalar_types = int, long, float, complex, SympyBasic


    @property
    def space(self):
        raise NotImplementedError(self.__class__.__name__)

    def n(self):
        return self
    
    def __neg__(self):
        return (-1)*self
    
    def adjoint(self):
        return Adjoint.create(self)
    
    conjugate = adjoint
    dag = adjoint


    def to_qutip(self, full_space = None):
        raise NotImplementedError(str(self.__class__))

    def __add__(self, other):
        if isinstance(other, Operator.scalar_types):
            return OperatorPlus.create(self, other * IdentityOperator)
        elif isinstance(other, Operator):
            return OperatorPlus.create(self, other)
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Operator.scalar_types):
            return ScalarTimesOperator.create(other, self)
        elif isinstance(other, Operator):
            return OperatorTimes.create(self, other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Operator.scalar_types):
            return ScalarTimesOperator.create(other, self)
        return NotImplemented

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __div__(self, other):
        if isinstance(other, Operator.scalar_types):
            return self * (sympyOne/other)
        return NotImplemented


@check_signature
class OperatorSymbol(Operator, Operation):
    signature = str, HilbertSpace

    def __str__(self):
        return self.operands[0]

    def tex(self):
        return "{" + self.operands[0] + "}"

    def __repr__(self):
        return "OperatorSymbol({},{})".format(*map(repr,self.operands))

            
    def to_qutip(self, full_space = None):
        raise AlgebraError("Cannot convert operator symbol to representation matrix. Substitute first.")
        
    @property
    def space(self):
        return self.operands[1]


@singleton
class IdentityOperator(Operator):


    def tex(self):
        return "1"

    def __repr__(self):
        return "IdentityOperator"

    @property
    def space(self):
        return TrivialSpace

    def adjoint(self):
        return self
        
    def to_qutip(self, full_space):
        return qutip.tensor(*[qutip.qeye(s.dimension) for s in full_space.local_factors()])
    
    def mathematica(self):
        return "IdentityOperator"



from scipy.sparse import csr_matrix

@singleton
class OperatorZero(Operator):

    @property
    def space(self):
        return TrivialSpace

    def adjoint(self):
        return self

    def to_qutip(self, full_space):
        return qutip.tensor(*[qutip.Qobj(csr_matrix((), (s.dimension, s.dimension))) for s in full_space.local_factors()])


class LocalOperator(Operator, Operation):

    @property
    def space(self):
        return self.operands[0]


@check_signature
class Create(LocalOperator):
    signature = LocalSpace,

@check_signature
class Destroy(LocalOperator):
    signature = LocalSpace,


@check_signature
class LocalSigma(LocalOperator):
    signature = LocalSpace, (int, str), (int, str)

LocalProjector = lambda spc, state: LocalSigma.create(spc, state, state)

        

def Z(local_space, states = ("h", "g")):
    h, g = states
    return LocalProjector(local_space, h) - LocalProjector(local_space, g)

def X(local_space, states = ("h", "g")):
    h, g = states
    return LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h)

def Y(local_space, states = ("h", "g")):
    h, g = states
    return I * (-LocalSigma(local_space, h, g) + LocalSigma(local_space, g, h))




class OperatorOperation(Operator, Operation):
    signature = Operator,

    @property
    def space(self):
        return prod((o.space for o in self.operands), TrivialSpace)

    def n(self):
        return self.__class__.apply_with_rules(*map(n, self.operands))


u = wc("u", head = Operator.scalar_types)
v = wc("v", head = Operator.scalar_types)

A = wc("A", head = Operator)
B = wc("B", head = Operator)

ls = wc("ls", head = LocalSpace)
ra = wc("ra", head = (int, str))
rb = wc("rb", head = (int, str))
rc = wc("rc", head = (int, str))
rd = wc("rd", head = (int, str))


@flat
@orderless
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class OperatorPlus(OperatorOperation):
    neutral_element = OperatorZero

    @classmethod
    def order_key(cls, a):
        if isinstance(a, ScalarTimesOperator):
            return Operation.order_key(a.operands[1]), a.operands[0]
        return Operation.order_key(a), 1



    binary_rules = [
    ]
        
    def to_qutip(self, full_space = None):
        if full_space == None:
            full_space = self.space
        assert self.space <= full_space
        return sum((op.to_qutip(full_space) for op in self.operands), OperatorZero)
        



@flat
@orderless
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class OperatorTimes(OperatorOperation):
    neutral_element = IdentityOperator

    class OperatorOrderKey(object):
        def __init__(self, space):
            self.full = False
            if isinstance(space, LocalSpace):
                self.local_spaces = {space.operands,}
            elif space is TrivialSpace:
                self.local_spaces = set(())
            elif space is FullSpace:
                self.full = True
            else:
                assert isinstance(space, ProductSpace)
                self.local_spaces = {s.operands for s in space.operands}

        def __lt__(self, other):
            if self.full or len(self.local_spaces & other.local_spaces):
                return False
            return tuple(self.local_spaces) < tuple(other.local_spaces)

        def __gt__(self, other):
            if self.full or len(self.local_spaces & other.local_spaces):
                return False
            return tuple(self.local_spaces) > tuple(other.local_spaces)

        def __eq__(self, other):
            return self.full or len(self.local_spaces & other.local_spaces) > 0

    @classmethod
    def order_key(cls, a):
        return cls.OperatorOrderKey(a.space)

    binary_rules = []

    @classmethod
    def create(cls, *ops):
        if any(o == OperatorZero for o in ops):
            return OperatorZero
        return cls(*ops)


            
    def to_qutip(self, full_space = None):
        if any(len(op.space) > 1 for op in self.operands):
            return self.expand().to_qutip(full_space)
        if full_space == None or full_space == self.space:
            if len(self.space) == 1: # if only operands from same space
                return product((op.to_qutip(self.space) for op in self.operands), 1)
            else: # otherwise group by space and return tensor product
                ops_by_space = operands_by_space(self.operands)
                return qutip.tensor(*[OperatorTimes(*ops_by_space[HilbertSpace((hspc,))]).to_qutip() for hspc in self.space])
        else:
            return self.extend_to_space(full_space).to_qutip()
    

def operands_by_space(operands):
    ret_dict = {}
    spaces = map(space, operands)
#    full_space = set_union(spaces)
    assert all(len(spc) == 1 for spc in spaces)
    for spc in spaces:
        space_ops = filter(lambda op: op.space == spc, operands)
        if space_ops:
            ret_dict[spc] = space_ops
    return ret_dict

@match_replace
@check_signature
class ScalarTimesOperator(Operator, Operation):
    signature = Operator.scalar_types, Operator
    rules = [
        ((1, A), lambda A: A),
        ((0, A), lambda A: OperatorZero),
        ((u, OperatorZero), lambda u: OperatorZero),
    ]
    
    @property
    def space(self):
        return self.operands[1].space
    
    def n(self):
        # print complex(self.coeff)
        return complex(self.operands[0]) * self.operands[1]
        
    def __str__(self):
        coeff, term = self.operands

        if coeff == -1:
            return "(-%s)" % (term,)

        if is_number(coeff) and not isinstance(coeff, complex) and coeff < 0:
            return "(%g) * %s" % (coeff, term)

        return "%s * %s" % (coeff, term)

    def tex(self):
        coeff, term = self.operands
        if term == IdentityOperator:
            return tex(coeff)
        if coeff == -1:
            return "(-%s)" % tex(term)
        return "(%s)  %s" % (tex(coeff), tex(term))
#


    def to_qutip(self, full_space = None):
        return complex(self.coeff) * self.term.to_qutip(full_space)

ScalarTimesOperator.rules.append(((u, ScalarTimesOperator(v, A)), lambda u, v, A: (u * v) * A))
OperatorPlus.binary_rules += [
    ((ScalarTimesOperator(u,A), ScalarTimesOperator(v,A)), lambda u,v,A: (u+v)*A),
    ((ScalarTimesOperator(u,A), A), lambda u,A: (u+1)*A),
    ((A, ScalarTimesOperator(v,A)), lambda v,A: (1+v)*A),
    ((A,A), lambda A: 2*A),
]
OperatorTimes.binary_rules += [
    ((ScalarTimesOperator(u,A), B), lambda u, A, B: u * (A * B)),

    ((A, ScalarTimesOperator(u,B)), lambda A, u, B: u * (A * B)),

    ((LocalSigma(ls, ra, rb), LocalSigma(ls, rc, rd)),
        lambda ls, ra, rb, rc, rd: LocalSigma(ls, ra, rd)
            if rb == rc else OperatorZero),

    ((Create(ls), LocalSigma(ls, rc, rd)),
        lambda ls, rc, rd: sqrt(rc + 1) * LocalSigma(ls, rc + 1, rd)),

    ((Destroy(ls), LocalSigma(ls, rc, rd)),
        lambda ls, rc, rd: sqrt(rc) * LocalSigma(ls, rc - 1, rd)),

    ((LocalSigma(ls, rc, rd), Destroy(ls)),
        lambda ls, rc, rd: sqrt(rd + 1) * LocalSigma(ls, rc, rd + 1)),

    ((LocalSigma(ls, rc, rd), Create(ls)),
        lambda ls, rc, rd: sqrt(rd) * LocalSigma(ls, rc, rd - 1)),

    ((Destroy(ls),Create(ls)),
        lambda ls: IdentityOperator + Create(ls) * Destroy(ls)),
]


A_plus = wc("A", head = OperatorPlus)
A_times = wc("A", head = OperatorTimes)

@check_signature
@match_replace
class Adjoint(OperatorOperation):

    rules = [
        ((ScalarTimesOperator(u, A),), lambda u, A: conjugate(u) * A.adjoint()),
        ((A_plus,), lambda A: OperatorPlus.create(*[o.adjoint() for o in A.operands])),
        ((A_times,), lambda A: OperatorTimes.create(*[o.adjoint() for o in A.operands[::-1]])),
    ]

    def to_qutip(self, full_space = None):
        return qutip.dag(self.operands[0].to_qutip(full_space))

    def tex(self):
        return "\left(" + self.operands[0].tex() + r"\right)^\dagger"

Adjoint.rules += [
    ((Adjoint(A),), lambda A: A),
    ((Create(ls),), lambda ls: Destroy(ls)),
    ((Destroy(ls),), lambda ls: Create(ls)),
    ((LocalSigma(ls, ra, rb),), lambda ls, ra, rb: LocalSigma(ls, rb, ra)),
]